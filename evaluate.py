import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from utils.preprocessing import compute_mean_std
from data import MriKneeDataset
from accelerate import Accelerator


def dice_coefficient(pred, target):
    smooth = 1e-8
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


if __name__ == "__main__":
    # Initialize model with pre-computed mean and std
    model = Unet(
        dim=16, dim_mults=(1, 2, 4, 8, 16), norm_mean=73.6998, norm_std=52.7535
    )
    image_size = 384

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=100,
        eval=True,
    )

    # Load the checkpoint
    checkpoint_path = "./results/checkpoints/last_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("mps"))
        diffusion.load_state_dict(checkpoint["model"])
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Initialize Accelerator for mixed precision if available
    accelerator = Accelerator()
    model = accelerator.prepare(diffusion)
    model.eval()

    # Load test dataset
    test_dataset = MriKneeDataset(
        "./data/splitted/test",
        "./data/splitted/test_masks",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    # Evaluation
    total_dice = 0.0
    total_jaccard = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Evaluating"):
            raw = data[0].to(accelerator.device).unsqueeze(0)  # Add batch dimension
            true_mask = data[1].to(accelerator.device)

            # Get model prediction
            pred_mask = model.sample(raw=raw, batch_size=1)

            # Convert pred_mask to one-hot encoded format
            pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
            pred_mask = torch.nn.functional.one_hot(
                pred_mask, num_classes=true_mask.shape[1]
            )
            pred_mask = (
                pred_mask.permute(0, 4, 1, 2, 3).squeeze(0).squeeze(1)
            )  # Convert to (C, H, W)

            # Flatten the masks for metric calculation
            pred_mask_flat = pred_mask.cpu().numpy().ravel()
            true_mask_flat = true_mask.cpu().numpy().ravel()

            dice = dice_coefficient(
                torch.tensor(pred_mask_flat), torch.tensor(true_mask_flat)
            )
            jaccard = jaccard_score(
                true_mask_flat,
                pred_mask_flat,
                average="macro",  # Using macro average for multiclass
            )

            total_dice += dice.item()
            total_jaccard += jaccard
            num_samples += 1

    avg_dice = total_dice / num_samples
    avg_jaccard = total_jaccard / num_samples

    print(f"Average Dice Coefficient: {avg_dice}")
    print(f"Average Jaccard Index: {avg_jaccard}")
