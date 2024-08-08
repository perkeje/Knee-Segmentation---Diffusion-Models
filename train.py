import sys
import os


# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.gaussian_diffusion import GaussianDiffusion
from experiments.trainer import Trainer
from unet.unet import Unet
from utils.preprocessing import calculate_class_weights, compute_mean_std

if __name__ == "__main__":
    mean, std = compute_mean_std("./data/splitted/train")
    print("Mean and std:")
    print(mean, std)
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    image_size = 384
    class_weights = calculate_class_weights("./data/splitted/train")
    print("Class weights:")
    print(class_weights)
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=100,
        class_weights=class_weights,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        train_segmentations_folder="./data/splitted/train_masks",
        train_images_folder="./data/splitted/train",
        test_segmentations_folder="./data/splitted/test_masks",
        test_images_folder="./data/splitted/test",
        batch_size=4,
        val_size=0.4,
        val_metric_size=4,
        lr=1e-4,
        epochs=500,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=20,
        es_patience=6,
        lr_patience=3,
        results_folder="./results",
        checkpoint_folder="./results/checkpoints",
        best_checkpoint="best_checkpoint",
        last_checkpoint="last_checkpoint",
        log="log.json",
    )

    trainer.train()
