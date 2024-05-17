import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.gaussian_diffusion import GaussianDiffusion
from experiments.trainer import Trainer
from unet.unet import Unet
from utils.preprocessing import compute_mean_std

if __name__ == "__main__":
    mean, std = compute_mean_std("./data/splitted/train")
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
        val_images=2,
        train_lr=1e-4,
        epochs=100,
        ema_update_every=160,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=10,
        results_folder="./results",
        amp=False,
        patience=10,
        reduce_lr_patience=5,
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint.pt",
        last_checkpoint="last_checkpoint.pt",
        loss_log="loss_log.json",
    )

    trainer.train()
