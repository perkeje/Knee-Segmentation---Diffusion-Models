import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import utils
from tqdm.auto import tqdm
from accelerate import Accelerator
from pathlib import Path
from data import MriKneeDataset
from utils import cycle
from ema_pytorch import EMA

from utils.postprocessing import apply_argmax_and_coloring


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
        # val_images=20,
        train_lr=1e-4,
        epochs=500,
        ema_update_every=2,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=20,
        results_folder="./results",
        amp=True,
        # patience=10,
        # reduce_lr_patience=5,
        checkpoint_folder="./results/checkpoints",
        # best_checkpoint="best_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        loss_log="loss_log.json",
    ):
        super().__init__()

        self.accelerator = Accelerator(mixed_precision="fp16" if amp else "no")

        self.model = diffusion_model.to(self.accelerator.device)
        self.model = self.accelerator.prepare(self.model)

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

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
        # self.val_ds, self.test_ds = random_split(
        #     self.test_ds, [val_images, len(self.test_ds) - val_images]
        # )
        # print(
        #     "Training size: "
        #     + str(len(self.ds))
        #     + "\nValidation size: "
        #     + str(len(self.val_ds))
        # )
        self.train_dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        # self.val_dl = DataLoader(
        #     self.val_ds,
        #     batch_size=train_batch_size,
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=4,
        # )

        self.train_dl = self.accelerator.prepare(self.train_dl)
        # self.val_dl = self.accelerator.prepare(self.val_dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.scaler = GradScaler()

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # checkpoint folder
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True)

        # self.best_checkpoint_path = self.checkpoint_folder / best_checkpoint
        self.last_checkpoint_path = self.checkpoint_folder / last_checkpoint
        self.loss_log_path = self.checkpoint_folder / loss_log

        # step counter state
        self.step = 0

        # EMA
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # Scheduler and early stopping
        self.scheduler = StepLR(self.opt, step_size=int(0.05 * self.epochs), gamma=0.95)
        # self.patience = patience
        # self.best_loss = float("inf")
        # self.no_improvement_epochs = 0

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
            # "best_loss": self.best_loss,
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
        # self.best_loss = data["best_loss"]
        self.loss_log = data["loss_log"]

    def save_loss_log(self):
        with open(self.loss_log_path, "w") as f:
            json.dump(self.loss_log, f)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        for epoch in range(self.step // self.ds.__len__(), self.epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            with tqdm(
                total=len(self.ds),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not accelerator.is_main_process,
            ) as epoch_pbar:
                for data in self.train_dl:
                    segmentation = data[1].to(device)
                    raw = data[0].to(device)
                    with accelerator.autocast():
                        loss = self.model(segmentation, raw)
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad()
                    self.step += 1
                    num_batches += 1
                    epoch_pbar.update(self.batch_size)
                    train_loss = total_loss / num_batches
                    epoch_pbar.set_postfix(
                        train_loss=train_loss, lr=self.scheduler.get_last_lr()
                    )

                # Validation
                # self.model.eval()
                # val_loss = 0.0
                # print("Validating...")
                # with torch.no_grad():
                #     for data in tqdm(self.val_dl):
                #         segmentation = data[1].to(device)
                #         raw = data[0].to(device)
                #         with accelerator.autocast():
                #             loss = self.model(segmentation, raw)
                #         val_loss += loss.item()
                # val_loss /= len(self.val_dl)
                self.scheduler.step()
                # print(val_loss)

            self.loss_log.append({"epoch": epoch + 1, "train_loss": train_loss})

            # print("Val loss: " + str(val_loss))

            # Save checkpoints
            if (epoch + 1) % self.save_and_sample_every == 0:
                self.ema.update()
                self.ema.ema_model.eval()

                with torch.no_grad():
                    milestone = (epoch + 1) // self.save_and_sample_every
                    validation_sample = self.test_ds[
                        0
                    ]  # Assuming the first validation sample is used
                    raw = validation_sample[0].unsqueeze(0).to(device)
                    sampled_images = self.ema.ema_model.sample(raw=raw, batch_size=1)

                    # Apply argmax and coloring
                    colored_sampled_images = (
                        apply_argmax_and_coloring(sampled_images) / 255.0
                    )

                    utils.save_image(
                        colored_sampled_images,
                        str(self.results_folder / f"sample-{milestone}.png"),
                    )

            # Save best and last checkpoints
            # if val_loss < self.best_loss:
            #     self.best_loss = val_loss
            #     self.save(self.best_checkpoint_path)
            #     self.no_improvement_epochs = 0
            # else:
            #     self.no_improvement_epochs += 1

            self.save(self.last_checkpoint_path)
            self.save_loss_log()

            # Early stopping
            # if self.no_improvement_epochs >= self.patience:
            #     print(f"Early stopping at epoch {epoch + 1}")
            #     break

        accelerator.print("Training complete!")
