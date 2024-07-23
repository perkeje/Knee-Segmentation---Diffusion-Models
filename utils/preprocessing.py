import nibabel as nib
import os
from tqdm import tqdm
import glob
import torch
import torchio as tio
import os


def load_mri(path):
    """Load MRI data from a file in a 3D tensor"""
    img = nib.load(path)
    return img.get_fdata()


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
        paths.extend([os.path.basename(p) for p in files])

    mean = 0.0
    std = 0.0
    M2 = 0.0
    count = 0

    print("Calculating dataset mean and std...")
    for path in tqdm(paths):
        img = tio.ScalarImage(os.path.join(adapt_dir, path)).data.squeeze()[80, :, :]
        img = img.to(dtype=torch.double).to('cuda')
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


def calculate_class_weights(device, adapt_dir, num_classes=6, exts=["nii", "nii.gz"]):
    paths = []
    for ext in exts:
        files = glob.glob(os.path.join(adapt_dir, "**", f"*.{ext}"), recursive=True)
        paths.extend([os.path.basename(p) for p in files])
    class_counts = torch.zeros(num_classes).to('cuda')

    print("Calculating class weights...")
    for path in tqdm(paths):
        seg = tio.LabelMap(os.path.join(adapt_dir, path)).data.squeeze()[80, :, :].to('cuda')
        for i in range(num_classes):
            class_counts[i] += torch.sum(seg == i)

    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    return class_weights.clone().detach()
