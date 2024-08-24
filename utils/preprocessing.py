import nibabel as nib
import os
from tqdm import tqdm
import glob
import torch
import torchio as tio


def load_mri(path):
    """Load MRI data from a file in a 3D tensor"""
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_mri(tensor, affine, header, name, save_dir):
    name = name + ".nii.gz"
    image = nib.Nifti1Image(tensor, affine, header)
    nib.save(image, os.path.join(save_dir, name))


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def compute_mean_std(adapt_dir, exts=["nii", "nii.gz"]):
    paths = []
    for ext in exts:
        files = glob.glob(os.path.join(adapt_dir, "**", f"*.{ext}"), recursive=True)
        paths.extend(files)

    mean = torch.tensor(0.0, dtype=torch.double)
    M2 = torch.tensor(0.0, dtype=torch.double)
    count = torch.tensor(0, dtype=torch.double)

    print("Calculating dataset mean and std...")
    for path in tqdm(paths):
        img = tio.ScalarImage(path).data.squeeze().flatten()
        img = img.to(dtype=torch.double)
        batch_count = img.numel()

        batch_mean = img.mean()
        delta = batch_mean - mean

        count += batch_count
        mean += delta * batch_count / count
        M2 += (
            img.var(unbiased=False) * batch_count
            + delta**2 * batch_count * (count - batch_count) / count
        )

    variance = M2 / count
    std = torch.sqrt(variance)

    return mean, std


def calculate_class_weights(adapt_dir, num_classes=6, exts=["nii", "nii.gz"]):
    paths = []
    for ext in exts:
        files = glob.glob(os.path.join(adapt_dir, "**", f"*.{ext}"), recursive=True)
        paths.extend([os.path.basename(p) for p in files])
    class_counts = torch.zeros(num_classes)

    print("Calculating class weights...")
    for path in tqdm(paths):
        seg = tio.LabelMap(os.path.join(adapt_dir, path)).data.squeeze()
        for i in range(num_classes):
            class_counts[i] += torch.sum(seg == i)

    class_weights = 1 / (class_counts / class_counts.sum())
    return class_weights
