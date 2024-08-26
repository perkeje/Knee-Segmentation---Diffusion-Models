import sys
import os
import torch
from tqdm import tqdm
from utils import load_class_weights, load_mean_std
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from accelerate import Accelerator
import torch.nn.functional as F
from utils.preprocessing import load_mri, save_mri
from unet.gaussian_blur import GaussianSmoothing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load model and predict on an image.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to save the output images"
    )
    parser.add_argument(
        "--output-name", type=str, required=True, help="Path to save the output images"
    )

    batch_size = 8

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
    smoothing = GaussianSmoothing(6, (7, 7, 7), 2, 3).to(accelerator.device)

    # Prepare the model with Accelerator
    model = accelerator.prepare(diffusion)

    # Load model checkpoint
    checkpoint_path = "./results/checkpoints/best_checkpoint"
    accelerator.load_state(checkpoint_path)
    model.eval()

    # Load the MRI and mask unsqueeze for channel (c,d,h,w)
    nifti = load_mri("data/splitted/test/9250756.nii.gz")
    img = torch.from_numpy(nifti[0]).unsqueeze(0).half()
    img = img.permute(1, 0, 2, 3).half()

    pred_volume = []
    with torch.no_grad():
        with accelerator.autocast():
            for i in tqdm(range(0, 160 // batch_size)):
                predicted = model.sample(
                    raw=img[i * batch_size : i * batch_size + batch_size, :, :, :]
                    .to(accelerator.device)
                    .float(),
                    batch_size=batch_size,
                    disable_bar=True,
                )
                pred_volume.append(predicted)

    pred_volume = torch.stack(pred_volume, dim=0)
    pred_volume = pred_volume.view(-1, 6, 384, 384).permute(1, 0, 2, 3)
    pred_volume = F.pad(pred_volume, (3, 3, 3, 3, 3, 3), mode="reflect")
    pred_volume = smoothing(pred_volume.to(accelerator.device)).cpu()

    pred_volume = torch.argmax(pred_volume, dim=0).cpu()

    save_mri(pred_volume, nifti[1], nifti[2], args.output_name, args.output_dir)
