from random import random
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from .schedules import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)
from utils.preprocessing import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from utils import exists, extract, default, identity
from collections import namedtuple
from einops import reduce

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        class_weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        timesteps=500,
        sampling_timesteps=None,
        objective="pred_x0",
        beta_schedule="linear",
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.1,
        auto_normalize=True,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        eval=False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        # Just to set class weights to appropriate device
        self.first_loss = True
        self.class_weights = class_weights
        self.image_size = image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, segmentation, raw, t, clip_x_start=False):
        model_output = self.model(segmentation, raw, t)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(segmentation, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(segmentation, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(segmentation, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(segmentation, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, segmentation, raw, t, x_self_cond=None, clip_denoised=True
    ):
        preds = self.model_predictions(segmentation, raw, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=segmentation, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def condition_mean(self, cond_fn, mean, variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # this fixes a bug in the official OpenAI implementation:
        # https://github.com/openai/guided-diffusion/issues/51 (see point 1)
        # use the predicted mean for the previous timestep to compute gradient
        gradient = cond_fn(mean, t, **guidance_kwargs)
        new_mean = mean.float() + variance * gradient.float()
        print("gradient: ", (variance * gradient.float()).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(
        self,
        segmentation,
        raw,
        t: int,
        x_self_cond=None,
        cond_fn=None,
        guidance_kwargs=None,
    ):
        b, *_, device = *segmentation.shape, segmentation.device
        batched_times = torch.full(
            (b,), t, device=segmentation.device, dtype=torch.long
        )
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            segmentation=segmentation,
            raw=raw,
            t=batched_times,
            x_self_cond=x_self_cond,
            clip_denoised=True,
        )
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(
                cond_fn,
                model_mean,
                variance,
                segmentation,
                batched_times,
                guidance_kwargs,
            )

        noise = torch.randn_like(segmentation) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(
        self, raw, shape, return_all_timesteps=False, cond_fn=None, guidance_kwargs=None
    ):
        batch, device = shape[0], self.betas.device

        segmentation = torch.randn(shape, device=device)
        segmentations = [segmentation]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling loop time step",
            total=self.num_timesteps,
            disable=eval
        ):
            self_cond = x_start if self.self_condition else None
            segmentation, x_start = self.p_sample(
                segmentation, raw, t, self_cond, cond_fn, guidance_kwargs
            )
            if return_all_timesteps and t % 100 == 0:
                segmentations.append(segmentation)

        ret = (
            segmentation
            if not return_all_timesteps
            else torch.stack(segmentations, dim=1)
        )

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(
        self, raw, shape, return_all_timesteps=False, cond_fn=None, guidance_kwargs=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        segmentation = torch.randn(shape, device=device)
        segmentations = [segmentation]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                segmentation, raw, time_cond, clip_x_start=True
            )

            segmentations.append(segmentation)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(segmentation)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        ret = img if not return_all_timesteps else torch.stack(segmentations, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(
        self,
        raw,
        batch_size=16,
        return_all_timesteps=False,
        cond_fn=None,
        guidance_kwargs=None,
    ):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            raw,
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            cond_fn=cond_fn,
            guidance_kwargs=guidance_kwargs,
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, raw, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, raw, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, raw, x_start, t, noise=None):
        if self.first_loss and self.class_weights.device != x_start.device:
            self.class_weights = self.class_weights.to(x_start.device)
            self.first_loss = False

        b, c, h, w = x_start.shape

        # Ensure the number of channels (c) matches the number of classes
        assert (
            c == 6
        ), "Number of channels (c) must be equal to the number of classes (6)"

        noise = default(noise, lambda: torch.randn_like(x_start))

        # Sample noisy image
        segmentation = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Self-conditioning (if applicable)
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(segmentation, raw, t).pred_x_start
                x_self_cond.detach_()

        # Predict and take gradient step
        model_out = self.model(segmentation, raw, t)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Calculate individual channel losses
        channel_losses = F.mse_loss(
            model_out, target, reduction="none"
        )  # Shape: (b, c, h, w)

        # Reshape the weights to (1, c, 1, 1)
        reshaped_class_weights = self.class_weights.view(1, c, 1, 1)

        # Apply weights to each channel loss
        weighted_loss = channel_losses * reshaped_class_weights

        # Reduce loss (e.g., mean across channels, height, and width)
        loss = reduce(weighted_loss, "b c h w -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, segmentation, raw, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *segmentation.shape,
            segmentation.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        segmentation = self.normalize(segmentation)
        return self.p_losses(raw, segmentation, t, *args, **kwargs)
