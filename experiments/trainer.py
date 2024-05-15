import os
import math
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torchvision import utils
from tqdm.auto import tqdm
from accelerate import Accelerator
from pathlib import Path
from data import MriKneeDataset
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from utils import cycle, has_int_squareroot, num_to_groups
from ema_pytorch import EMA

from utils.postprocessing import apply_argmax_and_coloring
from utils.preprocessing import compute_mean_std


class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_segmentations_folder,
        train_images_folder,
        test_segmentations_folder,
        test_images_folder,
        *,
        train_batch_size=16,
        val_split=0.4,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        epochs=1000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=20,
        results_folder="./results",
        amp=True,
        patience=15,
        reduce_lr_patience=10,
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        loss_log="loss_log.json",
    ):
        super().__init__()

        self.accelerator = Accelerator(mixed_precision="fp16" if amp else "no")

        self.model = diffusion_model.to(self.accelerator.device)
        self.model = self.accelerator.prepare(self.model)

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.epochs = epochs
        self.train_lr = train_lr
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.ds = MriKneeDataset(
            train_images_folder,
            train_segmentations_folder,
        )

        self.test_ds = MriKneeDataset(
            test_images_folder,
            test_segmentations_folder,
        )

        test, self.val_ds = random_split(self.test_ds, [1 - val_split, val_split])
        print(
            "Training size: "
            + str(len(self.ds))
            + "\nValidation size: "
            + str(len(self.val_ds))
        )
        self.train_dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

        self.train_dl = self.accelerator.prepare(self.train_dl)
        self.val_dl = self.accelerator.prepare(self.val_dl)
        self.train_dl = cycle(self.train_dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.scaler = GradScaler()

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # checkpoint folder
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_folder / best_checkpoint
        self.last_checkpoint_path = self.checkpoint_folder / last_checkpoint
        self.loss_log_path = self.checkpoint_folder / loss_log

        # step counter state
        self.step = 0

        # EMA
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # Scheduler and early stopping
        self.scheduler = ReduceLROnPlateau(
            self.opt, patience=reduce_lr_patience, verbose=True
        )
        self.patience = patience
        self.best_loss = float("inf")
        self.no_improvement_epochs = 0

        # Loss log
        self.loss_log = []

        # Resume training if last checkpoint exists
        if self.last_checkpoint_path.exists():
            self.load(self.last_checkpoint_path)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "no_improvement_epochs": self.no_improvement_epochs,
            "loss_log": self.loss_log,
        }

        torch.save(data, str(path))

    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(path), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.scaler.load_state_dict(data["scaler"])
        self.best_loss = data["best_loss"]
        self.no_improvement_epochs = data["no_improvement_epochs"]
        self.loss_log = data["loss_log"]

    def save_loss_log(self):
        with open(self.loss_log_path, "w") as f:
            json.dump(self.loss_log, f)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.epochs,
            disable=not accelerator.is_main_process,
        ) as pbar:
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0
                num_batches = 0

                for _ in range(len(self.train_dl) // self.gradient_accumulate_every):
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.train_dl)
                        segmentation = data[1].to(device)
                        raw = data[0].to(device)
                        with autocast():
                            loss = self.model(segmentation, raw)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss)

                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad()
                    self.step += 1
                    num_batches += 1

                train_loss = total_loss / num_batches

                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data in self.val_dl:
                        segmentation = data[1].to(device)
                        raw = data[0].to(device)
                        with autocast():
                            loss = self.model(segmentation, raw)
                        val_loss += loss.item()
                val_loss /= len(self.val_dl)

                self.loss_log.append(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                )
                self.scheduler.step(val_loss)

                pbar.set_description(
                    f"Epoch {epoch+1}/{self.epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
                )
                pbar.update(1)

                # Save checkpoints
                if (epoch + 1) % self.save_and_sample_every == 0:
                    self.ema.update()
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        milestone = (epoch + 1) // self.save_and_sample_every
                        validation_sample = self.val_ds[80]
                        raw = validation_sample[0].unsqueeze(0).to(device)
                        sampled_images = self.ema.ema_model.sample(
                            raw=raw, batch_size=1
                        )

                        # Apply argmax and coloring
                        colored_sampled_images = apply_argmax_and_coloring(
                            sampled_images
                        )

                        utils.save_image(
                            colored_sampled_images,
                            str(self.results_folder / f"sample-{milestone}.png"),
                        )

                # Save best and last checkpoints
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save(self.best_checkpoint_path)
                    self.no_improvement_epochs = 0
                else:
                    self.no_improvement_epochs += 1

                self.save(self.last_checkpoint_path)
                self.save_loss_log()

                # Early stopping
                if self.no_improvement_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        accelerator.print("Training complete")


if __name__ == "__main__":
    mean, std = compute_mean_std("./data/splitted/trai")
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    image_size = 384
    diffusion = GaussianDiffusion(
        model, image_size=image_size, timesteps=1000  # number of steps
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        train_segmentations_folder="./data/splitted/train_masks",
        train_images_folder="./data/splitted/train",
        test_segmentations_folder="./data/splitted/test_masks",
        test_images_folder="./data/splitted/test",
        train_batch_size=1,
        val_split=0.4,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        epochs=1000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=20,
        results_folder="./results",
        amp=True,
        patience=15,
        reduce_lr_patience=10,
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        loss_log="loss_log.json",
    )

    trainer.train()
