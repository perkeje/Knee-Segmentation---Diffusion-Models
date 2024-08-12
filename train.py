import sys
import os
import torch

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion.gaussian_diffusion import GaussianDiffusion
from experiments.trainer import Trainer
from unet.unet import Unet


def load_or_compute_mean_std(data_dir):
    mean_std_path = os.path.join(data_dir, "mean_std.pt")

    if os.path.exists(mean_std_path):
        mean_std = torch.load(mean_std_path)
        mean, std = mean_std["mean"], mean_std["std"]
    else:
        print("Mean and std not found. You need to calculate these with pretrain.py first.")
        sys.exit(1)

    return mean, std


def load_or_compute_class_weights(data_dir):
    class_weights_path = os.path.join(data_dir, "class_weights.pt")

    if os.path.exists(class_weights_path):
        class_weights = torch.load(class_weights_path)
    else:
        print("Class weights not found. You need to calculate these with pretrain.py first.")
        sys.exit(1)

    return class_weights


if __name__ == "__main__":
    params_dir = "./results/params"

    mean, std = load_or_compute_mean_std(params_dir)
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    class_weights = load_or_compute_class_weights(params_dir)

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
        lr=1e-4,
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
