import sys
import os
from sklearn.metrics import f1_score, jaccard_score
import torch
from pathlib import Path
from torchvision import utils
from tqdm import tqdm
from utils import load_class_weights, load_mean_std
from utils.postprocessing import apply_argmax_and_coloring
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from accelerate import Accelerator
import torch.nn.functional as F
from utils.preprocessing import load_mri, save_mri

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load model and predict on an image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output images")

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

    # Prepare the model with Accelerator
    model = accelerator.prepare(diffusion)

    # Load model checkpoint
    checkpoint_path = "./results/checkpoints/last_checkpoint"
    accelerator.load_state(checkpoint_path)

    # Load the MRI and mask unsqueeze for channel (c,d,h,w)
    nifti = load_mri("data/splitted/test/9012867.nii.gz")
    img = torch.from_numpy(nifti[0]).unsqueeze(0).half()
    img = img.permute(1, 0, 2, 3).half()
    print("Input image shape:", img.shape)

    nifti = load_mri("data/splitted/test_masks/9012867.nii.gz")
    mask = torch.from_numpy(nifti[0])
    mask = F.one_hot(
        mask.to(dtype=torch.int64),
        num_classes=6,
    )
    mask = mask.permute(3, 0, 1, 2).half()
    print("Mask shape after one-hot encoding and permutation:", mask.shape)
    pred_volume = []

    with accelerator.autocast():
        for i in tqdm(range(0, 160 / batch_size)):
            predicted = model.sample(
                raw=img[i * batch_size : i * batch_size + batch_size, :, :, :]
                .to(accelerator.device)
                .float(),
                batch_size=batch_size,
                disable_bar=True,
            )
            pred_volume.append(predicted)

    pred_volume = torch.stack(pred_volume, dim=0)
    pred_volume = pred_volume.view(-1, 6, 384, 384).permute(1, 0, 384, 384)
    pred_volume = torch.argmax(pred_volume, dim=0).cpu()

    pred_mask_flat = pred_volume.flatten()
    true_mask_flat = torch.argmax(mask.float(), dim=0).flatten().cpu()

    f1 = f1_score(true_mask_flat, pred_mask_flat, average="macro")
    iou = jaccard_score(
        true_mask_flat,
        pred_mask_flat,
        average="macro",
    )
    print(f"F1 Score: {f1}")
    print(f"IoU: {iou}")

    save_mri(pred_volume, nifti[1], nifti[2], "example", ".")

    # Apply argmax and coloring to the predicted image and save it
    colored_output = apply_argmax_and_coloring(pred_volume[:, 80, :, :].unsqueeze(0))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(colored_output):
        utils.save_image(
            img / 255.0,
            output_dir / f"example_{i}.png",
        )
