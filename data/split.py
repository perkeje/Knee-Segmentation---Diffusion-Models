from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm
import typer

app = typer.Typer()


def split_data(raw_dirs, mask_dirs, save_dir, test_size=0.2):
    """Function for splitting .nii and .nii.gz data with masks that is splitted across multiple directories"""
    mask_dict = {}

    for dir_path in mask_dirs:
        for file in os.listdir(dir_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                mask_dict[file] = os.path.join(dir_path, file)

    all_files = []

    for dir_path in raw_dirs:
        for file in os.listdir(dir_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                raw_path = os.path.join(dir_path, file)
                if file in mask_dict:
                    all_files.append((raw_path, mask_dict[file]))
                else:
                    print(file + " does not have a mask, skipping...")

    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)

    print(
        "Splitted "
        + str(len(train_files))
        + " for training "
        + str(len(test_files))
        + " for testing"
    )
    train_dir = os.path.join(save_dir, "train")
    train_masks_dir = os.path.join(save_dir, "train_masks")
    test_dir = os.path.join(save_dir, "test")
    test_masks_dir = os.path.join(save_dir, "test_masks")
    os.mkdir(train_dir)
    os.mkdir(train_masks_dir)
    os.mkdir(test_dir)
    os.mkdir(test_masks_dir)

    print("Copying test files...")
    for test_file in tqdm(test_files):
        shutil.copy(test_file[0], os.path.join(test_dir, os.path.basename(test_file[0])))
        shutil.copy(test_file[1], os.path.join(test_masks_dir, os.path.basename(test_file[1])))

    print("Copying train files...")
    for train_file in tqdm(train_files):
        shutil.copy(train_file[0], os.path.join(train_dir, os.path.basename(train_file[0])))
        shutil.copy(
            train_file[1],
            os.path.join(train_masks_dir, os.path.basename(train_file[1])),
        )


@app.command()
def main(
    raw_dirs: list[str] = typer.Option(..., help="List of directories containing raw images."),
    mask_dirs: list[str] = typer.Option(..., help="List of directories containing mask images."),
    save_dir: str = typer.Option(..., help="Directory where the split datasets should be saved."),
    test_size: float = typer.Option(
        0.2, help="Proportion of the dataset to include in the test split (default: 0.2)."
    ),
):
    split_data(raw_dirs, mask_dirs, save_dir, test_size)


if __name__ == "__main__":
    app()
