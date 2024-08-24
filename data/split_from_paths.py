import os
import shutil
from tqdm import tqdm
import typer

app = typer.Typer()


def copy_files_based_on_paths(raw_dirs, mask_dirs, save_dir, paths_file):
    """Function for copying .nii and .nii.gz data and their masks based on a paths file."""
    mask_dict = {}

    # Build a dictionary of mask files
    for dir_path in mask_dirs:
        for file in os.listdir(dir_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                mask_dict[file] = os.path.join(dir_path, file)

    all_files = []

    # Build a list of raw image files with corresponding masks
    for dir_path in raw_dirs:
        for file in os.listdir(dir_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                raw_path = os.path.join(dir_path, file)
                if file in mask_dict:
                    all_files.append((raw_path, mask_dict[file]))
                else:
                    print(f"{file} does not have a corresponding mask, skipping...")

    # Read paths from paths_file and determine the split
    with open(paths_file, "r") as f:
        paths = f.readlines()

    train_files = []
    test_files = []

    for path in paths:
        path = path.strip()
        filename = os.path.basename(path)

        for raw_file, mask_file in all_files:
            if filename == os.path.basename(raw_file):
                if "/train/" in path:
                    train_files.append((raw_file, mask_file))
                elif "/test/" in path:
                    test_files.append((raw_file, mask_file))
                break

    print(
        f"Found {len(train_files)} train files and {len(test_files)} test files based on paths file."
    )

    # Directories to save the copied files
    train_dir = os.path.join(save_dir, "train")
    train_masks_dir = os.path.join(save_dir, "train_masks")
    test_dir = os.path.join(save_dir, "test")
    test_masks_dir = os.path.join(save_dir, "test_masks")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)

    # Copy the files to the respective directories
    print("Copying test files...")
    for test_file in tqdm(test_files):
        shutil.copy(test_file[0], os.path.join(test_dir, os.path.basename(test_file[0])))
        shutil.copy(test_file[1], os.path.join(test_masks_dir, os.path.basename(test_file[1])))

    print("Copying train files...")
    for train_file in tqdm(train_files):
        shutil.copy(train_file[0], os.path.join(train_dir, os.path.basename(train_file[0])))
        shutil.copy(train_file[1], os.path.join(train_masks_dir, os.path.basename(train_file[1])))


@app.command()
def main(
    raw_dirs: list[str] = typer.Option(..., help="List of directories containing raw images."),
    mask_dirs: list[str] = typer.Option(..., help="List of directories containing mask images."),
    save_dir: str = typer.Option(..., help="Directory where the split datasets should be saved."),
    paths_file: str = typer.Option(
        ..., help="Path to the paths.txt file that defines the train/test split."
    ),
):
    copy_files_based_on_paths(raw_dirs, mask_dirs, save_dir, paths_file)


if __name__ == "__main__":
    app()
