import sys
import os

import torch

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.gaussian_diffusion import GaussianDiffusion
from experiments.trainer import Trainer
from unet.unet import Unet
from utils.preprocessing import calculate_class_weights, compute_mean_std

if __name__ == "__main__":
    # mean, std = compute_mean_std("./data/splitted/train")
    model = Unet(dim=16, dim_mults=(1, 2, 4, 8, 16, 32), norm_mean=0, norm_std=1)
    image_size = 384
    # class_weights = calculate_class_weights("./data/splitted/train")
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        # class_weights=class_weights,  # number of steps
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        train_segmentations_folder="./data/splitted/train_masks",
        train_images_folder="./data/splitted/train",
        test_segmentations_folder="./data/splitted/test_masks",
        test_images_folder="./data/splitted/test",
        train_batch_size=2,
        # val_images=50,
        train_lr=1e-4,
        epochs=100,
        ema_update_every=2,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=10,
        results_folder="./results",
        amp=False,
        # patience=8,
        # reduce_lr_patience=4,
        checkpoint_folder="./results/checkpoints",
        # best_checkpoint="best_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        loss_log="loss_log.json",
    )

    trainer.train()
