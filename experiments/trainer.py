import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import utils
from tqdm.auto import tqdm
from accelerate import Accelerator
from pathlib import Path
from data import MriKneeDataset
import math
from utils.postprocessing import apply_argmax_and_coloring
from sklearn.metrics import jaccard_score, f1_score
from accelerate.utils import set_seed


class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_segmentations_folder,
        train_images_folder,
        test_segmentations_folder,
        test_images_folder,
        *,
        batch_size=4,
        val_size=0.4,
        val_metric_size=4,
        lr=1e-4,
        epochs=500,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=20,
        es_patience=10,
        lr_patience=5,
        results_folder="./results",
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint",
        last_checkpoint="last_checkpoint",
        log="log.json",
    ):
        super().__init__()
        set_seed(42)

        self.accelerator = Accelerator()

        self.save_and_sample_every = save_and_sample_every
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr * self.accelerator.num_processes
        self.image_size = diffusion_model.image_size
        self.val_metric_size = val_metric_size

        # dataset and dataloader
        self.train_ds = MriKneeDataset(
            train_images_folder,
            train_segmentations_folder,
        )

        test_ds = MriKneeDataset(
            test_images_folder,
            test_segmentations_folder,
        )
        self.val_ds, self.test_ds = random_split(
            test_ds, [val_size, 1 - val_size], torch.Generator()
        )
        self.accelerator.print(
            "Training size: "
            + str(len(self.train_ds))
            + "\nValidation size: "
            + str(len(self.val_ds))
        )
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4 * self.accelerator.num_processes,
        )

        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4 * self.accelerator.num_processes,
        )

        self.val_metric_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4 * self.accelerator.num_processes,
        )
        # optimizer
        self.optimizer = Adam(diffusion_model.parameters(), lr=self.lr, betas=adam_betas)

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # checkpoint folder
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_folder / best_checkpoint
        self.last_checkpoint_path = self.checkpoint_folder / last_checkpoint
        self.log_path = self.checkpoint_folder / log

        # step counter state
        self.step = 0

        # Scheduler and early stopping
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=lr_patience)
        self.es_patience = es_patience
        self.best_metric = 0
        self.no_improvement_epochs = 0

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dl,
            self.val_dl,
            self.val_metric_dl,
        ) = self.accelerator.prepare(
            diffusion_model,
            self.optimizer,
            self.scheduler,
            self.train_dl,
            self.val_dl,
            self.val_metric_dl,
        )

        # Loss log
        self.logs = []

        # Resume training if last checkpoint exists
        if self.last_checkpoint_path.exists():
            self.accelerator.load_state(self.last_checkpoint_path)

        if self.log_path.exists():
            self.load_log()
            self.best_metric = self.logs[-1]["best_metric"]
            self.step = self.logs[-1]["step"]

    def append_to_log(self, step, loss, val_loss, metric, best_metric):
        if not self.accelerator.is_local_main_process:
            return
        self.logs.append(
            {
                "step": step,
                "loss": loss,
                "val_loss": val_loss,
                "metric": metric,
                "best_metric": best_metric,
            }
        )
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)

    def load_log(self):
        with open(self.log_path, "r") as f:
            self.logs = json.load(f)

    def train(self):

        for epoch in range(
            self.step // (math.ceil(len(self.train_dl) // self.batch_size)), self.epochs
        ):
            self.model.train()

            with tqdm(
                total=len(self.train_dl),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                colour="green",
                disable=not self.accelerator.is_main_process,
            ) as train_bar:
                for data in self.train_dl:
                    segmentation = data[1]
                    raw = data[0]
                    with self.accelerator.autocast():
                        loss = self.model(segmentation, raw)

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.step += 1
                    train_bar.update(1)
                    train_bar.set_postfix(
                        train_loss=loss.item(),
                        lr=self.scheduler.get_last_lr() / 4,
                        gpu_lr=self.scheduler.get_last_lr(),
                    )
                train_bar.close()
                self.accelerator.wait_for_everyone()

                # Validation
                self.model.eval()
                val_loss = []
                f1_list = []
                iou_list = []
                with torch.no_grad():
                    with tqdm(
                        total=len(self.val_dl)
                        + self.val_metric_size
                        * 160
                        // self.batch_size
                        // self.accelerator.num_processes,
                        desc="/tValidation",
                        colour="blue",
                        disable=not self.accelerator.is_main_process,
                    ) as val_bar:
                        for data in self.val_dl:
                            segmentation = data[1]
                            raw = data[0]
                            with self.accelerator.autocast():
                                val_loss.append(self.model(segmentation, raw).item())
                            val_bar.update(1)

                        for i, data in enumerate(self.val_metric_dl):
                            segmentation = data[1]
                            raw = data[0]
                            with self.accelerator.autocast():
                                sampled = self.model.sample(
                                    raw=raw, batch_size=self.batch_size, disable_bar=True
                                )
                            pred_mask_flat = torch.argmax(sampled, dim=1).flatten()
                            true_mask_flat = torch.argmax(segmentation, dim=1).flatten()

                            f1 = f1_score(true_mask_flat, pred_mask_flat, average="micro")
                            iou = jaccard_score(
                                true_mask_flat,
                                pred_mask_flat,
                                average="micro",
                            )
                            f1_list.append(f1)
                            iou_list.append(iou)
                            val_bar.update(1)
                            if (
                                i
                                == self.val_metric_size
                                * 160
                                // self.batch_size
                                // self.accelerator.num_processes
                            ):
                                break

                self.accelerator.wait_for_everyone()
                val_loss = torch.tensor(val_loss)
                f1_list = torch.tensor(f1_list)
                iou_list = torch.tensor(iou_list)

                val_loss = self.accelerator.gather(val_loss)
                f1 = self.accelerator.gather(f1_list)
                iou = self.accelerator.gather(iou_list)

                if self.accelerator.is_main_process:
                    val_loss = val_loss.mean().item()
                    f1 = f1.mean().item()
                    iou = iou.mean().item()
                    train_bar.set_postfix(val_loss=val_loss, f1=f1, iou=iou),
                    metric = f1 + iou
                    if metric > self.best_metric:
                        self.best_metric = metric
                        self.accelerator.save_state(self.best_checkpoint_path)
                        self.no_improvement_epochs = 0
                    else:
                        self.no_improvement_epochs += 1

                    self.append_to_log(self.step, loss.item(), val_loss, metric, self.best_metric)
                    self.accelerator.save_state(self.last_checkpoint_path)

                self.scheduler.step(metric)

            # Save and sample
            if self.accelerator.is_main_process and (epoch + 1) % self.save_and_sample_every == 0:

                with torch.no_grad():
                    milestone = (epoch + 1) // self.save_and_sample_every
                    validation_sample = self.val_ds[0]
                    raw = validation_sample[0].unsqueeze(0)
                    with self.accelerator.autocast():
                        sampled_images = self.model.sample(raw=raw, batch_size=1)

                    # Apply argmax and coloring
                    colored_sampled_images = apply_argmax_and_coloring(sampled_images) / 255.0

                    utils.save_image(
                        colored_sampled_images,
                        str(self.results_folder / f"sample-{milestone}.png"),
                    )

            # Early stopping
            if self.accelerator.is_main_process and self.no_improvement_epochs >= self.es_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.accelerator.print("Training complete!")
