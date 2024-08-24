import sys
import os

from utils import load_mean_std, load_class_weights

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.gaussian_diffusion import GaussianDiffusion
from experiments.trainer import Trainer
from unet.unet import Unet


if __name__ == "__main__":
    params_dir = "./results/params"

    mean, std = load_mean_std(params_dir)
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    class_weights = load_class_weights(params_dir)

    diffusion = GaussianDiffusion(
        model,
        image_size=384,
        timesteps=100,
        class_weights=class_weights,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        train_segmentations_folder="./data/splitted/train_masks",
        train_images_folder="./data/splitted/train",
        test_segmentations_folder="./data/splitted/test_masks",
        test_images_folder="./data/splitted/test",
        batch_size=8,
        val_size=0.4,
        val_metric_size=4,
        lr=5e-5,
        epochs=250,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=5,
        es_patience=6,
        lr_patience=3,
        results_folder="./results",
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint",
        last_checkpoint="last_checkpoint",
        log="log.json",
    )

    trainer.train()
