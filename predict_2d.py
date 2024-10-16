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
import torch.nn.functional as F

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) )
    model = accelerator.prepare(diffusion)
    checkpoint_path = "./results/checkpoints/best_checkpoint"
    accelerator.load_state(checkpoint_path)
    model.eval()

    # Load the MRI and select the 80th slice (c, h, w)
    nifti = load_mri("data/splitted/test/9250756.nii.gz")
    img = torch.from_numpy(nifti[0]).unsqueeze(0).half()

    # utils.save_image(torch.from_numpy(nifti[0][79]) / 255.0, output_dir / f"{args.output_name}_original.png")

    nifti = load_mri("data/splitted/test_masks/9250756.nii.gz")
    mask = torch.from_numpy(nifti[0])[79,:,:].to(dtype=torch.int64)
    print(mask.shape)
    mask = F.one_hot(
        mask,
        num_classes=6,
    )
    mask = mask.permute(2, 0, 1).half().unsqueeze(0)
    utils.save_image(apply_argmax_and_coloring(mask) / 255.0, output_dir / f"{args.output_name}_original_mask.png")


    # Select the 80th slice
    slice_img = img[:, 79, :, :].unsqueeze(0).to(accelerator.device).float()
    # slice_mask = 
    # Predict the 80th slice
    with torch.no_grad():
        with accelerator.autocast():
            predicted = model.sample(
                raw=slice_img,
                batch_size=1,
                disable_bar=False,
                return_all_timesteps=True
            )
    # Apply argmax and coloring to the predicted image and save it
    colored_output = apply_argmax_and_coloring(predicted.squeeze(0))


    for i, sample in enumerate(colored_output):
        utils.save_image(
            sample / 255.0,
            output_dir / f"{args.output_name}_{i}.png",
        )


    print(f"Prediction for the 80th slice saved at {output_dir}")
