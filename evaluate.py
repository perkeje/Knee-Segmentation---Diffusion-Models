import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from data import MriKneeDataset3D
from accelerate import Accelerator
from sklearn.metrics import f1_score
from utils import load_mean_std, load_class_weights
from unet.gaussian_blur import GaussianSmoothing
import torch.nn.functional as F


if __name__ == "__main__":
    params_dir = "./results/params"
    accelerator = Accelerator()
    mean, std = load_mean_std(params_dir)
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8, 16), norm_mean=mean, norm_std=std)
    class_weights = load_class_weights(params_dir)

    diffusion = GaussianDiffusion(
        model,
        image_size=384,
        timesteps=100,
        class_weights=class_weights,
    )
    smoothing = GaussianSmoothing(6, (7, 7, 7), 2, 3)

    model = accelerator.prepare(diffusion)
    checkpoint_path = "./results/checkpoints/best_checkpoint"
    accelerator.load_state(checkpoint_path)

    # Load test dataset
    test_dataset = MriKneeDataset3D(
        "./data/splitted/test",
        "./data/splitted/test_masks",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4 * accelerator.num_processes
    )

    model, test_dataloader, smoothing = accelerator.prepare(model, test_dataloader, smoothing)

    f1 = []
    iou = []
    f1_smoothed = []
    iou_smoothed = []
    batch_size = 8
    with torch.no_grad():
        with accelerator.autocast():
            for data in tqdm(
                test_dataloader,
                desc="Evaluating:",
                disable=not accelerator.is_main_process,
                colour="green",
            ):

                raw = data[0].squeeze(0).permute(1, 0, 2, 3).half()
                true_mask = data[1].squeeze(0)

                pred_volume = []
                for i in range(0, 160 // batch_size):
                    predicted = model.sample(
                        raw=raw[i * batch_size : i * batch_size + batch_size, :, :, :]
                        .to(accelerator.device)
                        .float(),
                        batch_size=batch_size,
                        disable_bar=True,
                    )
                    pred_volume.append(predicted)

                pred_volume = torch.stack(pred_volume, dim=0)
                pred_volume = pred_volume.view(-1, 6, 384, 384).permute(1, 0, 2, 3)

                pred_flattened = torch.argmax(pred_volume, dim=0).flatten().cpu()
                true_mask = torch.argmax(true_mask, dim=0).flatten().cpu()

                dice = f1_score(true_mask, pred_flattened, average="macro")
                jaccard = jaccard_score(
                    true_mask,
                    pred_flattened,
                    average="macro",
                )

                f1.append(dice)
                iou.append(jaccard)

                pred_volume = F.pad(pred_volume, (3, 3, 3, 3, 3, 3), mode="reflect")
                pred_volume = smoothing(pred_volume)
                pred_flattened = torch.argmax(pred_volume, dim=0).flatten().cpu()

                dice = f1_score(true_mask, pred_flattened, average="macro")
                jaccard = jaccard_score(
                    true_mask,
                    pred_flattened,
                    average="macro",
                )

                f1_smoothed.append(dice)
                iou_smoothed.append(jaccard)

    accelerator.wait_for_everyone()
    f1 = torch.tensor(f1).to(accelerator.device)
    f1 = accelerator.gather(f1).mean().item()
    iou = torch.tensor(iou).to(accelerator.device)
    iou = accelerator.gather(iou).mean().item()
    f1_smoothed = torch.tensor(f1_smoothed).to(accelerator.device)
    f1_smoothed = accelerator.gather(f1_smoothed).mean().item()
    iou_smoothed = torch.tensor(iou_smoothed).to(accelerator.device)
    iou_smoothed = accelerator.gather(iou_smoothed).mean().item()

    print(f"Average Dice Coefficient: {f1}")
    print(f"Average Jaccard Index: {iou}")
    print(f"Average Dice Coefficient Smoothed: {f1_smoothed}")
    print(f"Average Jaccard Index Smoothed: {iou_smoothed}")
    with open("eval.txt", "w") as f:
        f.write(f"Average Dice Coefficient: {f1}\n")
        f.write(f"Average Jaccard Index: {iou}\n")
        f.write(f"Average Dice Coefficient Smoothed: {f1_smoothed}\n")
        f.write(f"Average Jaccard Index Smoothed: {iou_smoothed}\n")
