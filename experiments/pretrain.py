import os

import torch
from utils.preprocessing import calculate_class_weights, compute_mean_std
import typer

app = typer.Typer()


def save_results(mean, std, class_weights, results_save_dir):
    """Function to save the computed mean, std, and class weights as tensors."""
    os.makedirs(results_save_dir, exist_ok=True)

    mean_std_path = os.path.join(results_save_dir, "mean_std.pt")
    class_weights_path = os.path.join(results_save_dir, "class_weights.pt")

    torch.save({"mean": mean, "std": std}, mean_std_path)

    torch.save(class_weights, class_weights_path)

    print(f"Tensors saved in {results_save_dir}.")


@app.command()
def main(
    adapt_dir: str = typer.Option(..., help="Directory containing the dataset."),
    results_save_dir: str = typer.Option(..., help="Directory where the results should be saved."),
    num_classes: int = typer.Option(6, help="Number of classes for calculating class weights."),
    exts: list[str] = typer.Option(["nii", "nii.gz"], help="List of file extensions to include."),
):
    mean, std = compute_mean_std(adapt_dir, exts)

    class_weights = calculate_class_weights(adapt_dir, num_classes, exts)

    save_results(mean, std, class_weights, results_save_dir)


if __name__ == "__main__":
    app()
