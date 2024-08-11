import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from utils import default
from utils import exists
from torchvision.transforms import Normalize

from utils.preprocessing import normalize_to_neg_one_to_one


class NoSegmentatioStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, _):
        return image_features


class DynamicConditionalEncoding(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()

        self.layer_norm = nn.LayerNorm(feature_channels)

    def forward(self, image_features, segmentation_features):

        norm_image_features = self.layer_norm(image_features.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        norm_segmentation_features = self.layer_norm(
            segmentation_features.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        affinity = norm_image_features * norm_segmentation_features

        enhanced_features = affinity * image_features

        return enhanced_features


class FFParser(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()

        self.attentive_map = nn.Parameter(torch.randn(1, channels, height, width))

    def forward(self, features):

        fft_features = torch.fft.fft2(features)

        modulated_fft_features = fft_features * self.attentive_map

        modulated_features = torch.fft.ifft2(modulated_fft_features).real

        return modulated_features


class SegmentationStep(nn.Module):
    def __init__(self, feature_channels, height, width):
        super().__init__()
        self.dynamic_cond_enc = DynamicConditionalEncoding(feature_channels)
        self.ff_parser = FFParser(feature_channels, height, width)

    def forward(self, image_features, segmentation_features):
        enhanced_features = self.dynamic_cond_enc(image_features, segmentation_features)

        final_features = self.ff_parser(enhanced_features)

        return final_features


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[-1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        norm_mean,
        norm_std,
        img_size=384,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=6,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.mean = norm_mean
        self.std = norm_std
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv_segmentation = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        self.init_conv_raw = nn.Conv2d(1, init_dim, 7, padding=3)

        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.segmentation_downs = nn.ModuleList([])
        self.raw_downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.segmentation_downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )
        current_img_size = img_size
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.raw_downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            SegmentationStep(dim_in, current_img_size, current_img_size)
                            if not is_last
                            else NoSegmentatioStep()
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )
            current_img_size = current_img_size // 2

        mid_dim = dims[-1] * 2
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_last_conv = nn.Conv2d(mid_dim, dims[-1], 3, padding=1)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_out + dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, segmentation, raw, time):
        y = Normalize(self.mean, self.std)(raw)

        x = self.init_conv_segmentation(segmentation)
        y = self.init_conv_raw(y)

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.segmentation_downs:
            x = block1(x, t)
            # h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        for block1, block2, attn, segmentation_step, downsample in self.raw_downs:
            y = block1(y, t)
            # h.append(x)

            y = block2(y, t)
            y = attn(y)
            y = segmentation_step(y, h.pop(0))
            h.append(y)
            y = downsample(y)

        x = torch.cat((x, y), dim=1)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_last_conv(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        # x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)
        return normalize_to_neg_one_to_one(x)
