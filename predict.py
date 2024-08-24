import sys
import os
from sklearn.metrics import f1_score, jaccard_score
import torch
from pathlib import Path
from torchvision import utils
from tqdm import tqdm
from data.dataset import MriKneeDataset
from utils import load_class_weights, load_mean_std
from utils.postprocessing import apply_argmax_and_coloring
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from accelerate.utils import set_seed
import torch.nn.functional as F

from utils.preprocessing import load_mri

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define color palette for the segmentation classes
COLORS = torch.tensor(
    [
        [0, 0, 0],  # Background (Black)
        [255, 0, 0],  # Class 1 (Red)
        [0, 255, 0],  # Class 2 (Green)
        [0, 0, 255],  # Class 3 (Blue)
        [255, 255, 0],  # Class 4 (Yellow)
        [255, 0, 255],  # Class 5 (Magenta)
    ],
    dtype=torch.uint8,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load model and predict on an image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output images")

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

    # Prepare the model with Accelerator
    model = accelerator.prepare(diffusion)

    # Load model checkpoint
    checkpoint_path = "./results/checkpoints/best_checkpoint"
    accelerator.load_state(checkpoint_path)

    # Create dataset and get the image and mask
    nifti = load_mri("data/splitted/test/9002430.nii.gz")
    img = torch.from_numpy(nifti[0])
    print(img.shape)
    nifti = load_mri("data/splitted/test_masks/9002430.nii.gz")
    mask = torch.from_numpy(nifti[0])
    mask = F.one_hot(
        mask.to(dtype=torch.int64),
        num_classes=6,
    )
    mask = mask.permute(3, 0, 1, 2).half()
    pred_volume = []
    with accelerator.autocast():
        for slice in tqdm(range(0, img.shape[0])):
            predicted = model.sample(
                raw=img[slice, :, :].half().unsqueeze(0).to(accelerator.device).float(),
                batch_size=1,
                disable_bar=True,
            )

    pred_volume = torch.stack(pred_volume, dim=0)
    pred_volume = pred_volume.permute(1, 2, 0, 3, 4).squeeze(0)
    print(pred_volume.shape)
    pred_mask_flat = torch.argmax(pred_volume, dim=0).flatten().cpu()
    true_mask_flat = torch.argmax(mask.float(), dim=0).flatten().cpu()

    f1 = f1_score(true_mask_flat, pred_mask_flat, average="macro")
    iou = jaccard_score(
        true_mask_flat,
        pred_mask_flat,
        average="macro",
    )
    print(f1)
    print(iou)
    # Apply argmax and coloring to the predicted image and save it
    colored_output = apply_argmax_and_coloring(pred_volume[80])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(colored_output):
        utils.save_image(
            img / 255.0,
            output_dir / f"example_{i}.png",
        )
