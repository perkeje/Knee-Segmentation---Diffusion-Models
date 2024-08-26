import random
import glob
import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn.functional as F
import torchio as tio
import os


class MriKneeDataset(data.Dataset):
    def __init__(
        self,
        raw_dir,
        masks_dir,
        channels=6,
        transform=False,
        exts=["nii", "nii.gz"],
    ):
        super().__init__()
        self.raw_dir = raw_dir
        self.masks_dir = masks_dir
        self.channels = channels
        self.exts = exts
        self.paths = []
        for ext in exts:
            files = glob.glob(os.path.join(raw_dir, "**", f"*.{ext}"), recursive=True)
            self.paths.extend([os.path.basename(p) for p in files])

        random.shuffle(self.paths)

        self.latest_name = ""
        self.transform = transform
        self.transformations = T.Compose(
            [
                tio.RandomAffine(degrees=(0, 5, 5)),
            ]
        )

    def __len__(self):
        return len(self.paths) * 160

    def __getitem__(self, index):
        file_index = index // 160
        slice_index = index % 160
        path = self.paths[file_index]
        if not path.__eq__(self.latest_name):
            self.latest_name = path
            self.latest_subject = tio.Subject(
                raw_img=tio.ScalarImage(os.path.join(self.raw_dir, self.latest_name)),
                mask_img=tio.LabelMap(os.path.join(self.masks_dir, self.latest_name)),
            )
            if self.transform:
                self.latest_subject = self.transformations(self.latest_subject)

        raw_img = self.latest_subject.raw_img.data.squeeze()[slice_index, :, :]
        raw_img = raw_img.unsqueeze(0).half()
        mask = F.one_hot(
            self.latest_subject.mask_img.data.squeeze().to(dtype=torch.int64)[slice_index, :, :],
            num_classes=6,
        )
        mask = mask.permute(2, 0, 1).half()

        return raw_img, mask, slice_index


class MriKneeDataset3D(data.Dataset):
    def __init__(
        self,
        raw_dir,
        masks_dir,
        channels=6,
        transform=False,
        exts=["nii", "nii.gz"],
    ):
        super().__init__()
        self.raw_dir = raw_dir
        self.masks_dir = masks_dir
        self.channels = channels
        self.exts = exts
        self.paths = []
        for ext in exts:
            files = glob.glob(os.path.join(raw_dir, "**", f"*.{ext}"), recursive=True)
            self.paths.extend([os.path.basename(p) for p in files])

        random.shuffle(self.paths)

        self.transform = transform
        self.transformations = T.Compose(
            [
                tio.RandomAffine(degrees=(0, 5, 5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        name = self.paths[index]
        subject = tio.Subject(
            raw_img=tio.ScalarImage(os.path.join(self.raw_dir, name)),
            mask_img=tio.LabelMap(os.path.join(self.masks_dir, name)),
        )
        if self.transform:
            subject = self.transformations(subject)

        raw_img = subject.raw_img.data.half()
        mask = F.one_hot(
            subject.mask_img.data.squeeze().to(dtype=torch.int64),
            num_classes=6,
        )
        mask = mask.permute(3, 0, 1, 2).half()

        return raw_img, mask
