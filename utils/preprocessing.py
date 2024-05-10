import nibabel as nib
import os
from tqdm import tqdm
import glob
import torch
import torchio as tio


def load_mri(path):
    """Load MRI data from a file in a 3D tensor"""
    img = nib.load(path)
    return img.get_fdata()


def save_mri(tensor, affine, header, name, save_dir):
    name = name + ".nii.gz"
    image = nib.Nifti1Image(tensor, affine, header)
    nib.save(image, os.path.join(save_dir, name))


def compute_mean_std(adapt_dir, exts=["nii", "nii.gz"]):
    paths = []
    for ext in exts:
        files = glob.glob(os.path.join(adapt_dir, "**", f"*.{ext}"), recursive=True)
        paths.extend([os.path.basename(p) for p in files])

    mean = 0.0
    std = 0.0
    M2 = 0.0
    count = 0

    print("Calculating dataset mean and std:")
    for path in tqdm(paths):
        img = tio.ScalarImage(os.path.join(adapt_dir, path)).data.squeeze()
        img = img.to(dtype=torch.double)
        img_mean = img.mean()
        img_M2 = img.var()
        delta = img_mean - mean
        mean += delta / (count + 1)
        M2 += img_M2 + delta**2 * count / (count + 1)
        count += 1

    std = (M2 / count) ** 0.5
    print("MEAN: " + str(mean))
    print("STD: " + str(std))

    return mean, std
