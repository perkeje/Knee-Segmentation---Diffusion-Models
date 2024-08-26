import sys
import os
import torch
from pathlib import Path
from torchvision import utils
from utils import load_class_weights, load_mean_std
from utils.postprocessing import apply_argmax_and_coloring
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from accelerate import Accelerator
from utils.preprocessing import load_mri

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load model and predict on an image.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the output images"
    )
    parser.add_argument(
        "--output-name", type=str, required=True, help="Name to save the output image"
    )

    args = parser.parse_args()
    accelerator = Accelerator()

    # Load model parameters from pretrain.py
    params_dir = "./results/params"
    mean, std = load_mean_std(params_dir)
    class_weights = load_class_weights(params_dir)

    # Initialize your model here
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    image_size = 384
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=100,
        class_weights=class_weights,
    )

    model = accelerator.prepare(diffusion)
    checkpoint_path = "./results/checkpoints/best_checkpoint"
    accelerator.load_state(checkpoint_path)
    model.eval()

    # Load the MRI and select the 80th slice (c, h, w)
    nifti = load_mri("data/splitted/test/9012867.nii.gz")
    img = torch.from_numpy(nifti[0]).unsqueeze(0).half()

    # Select the 80th slice
    slice_img = img[:, 79, :, :].unsqueeze(0).to(accelerator.device).float()

    # Predict the 80th slice
    with torch.no_grad():
        with accelerator.autocast():
            predicted = model.sample(
                raw=slice_img,
                batch_size=1,
                disable_bar=False,
            )

    # Apply argmax and coloring to the predicted image and save it
    colored_output = apply_argmax_and_coloring(predicted)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.save_image(
        colored_output.squeeze(0) / 255.0,
        output_dir / f"{args.output_name}.png",
    )

    print(f"Prediction for the 80th slice saved at {output_dir}")
