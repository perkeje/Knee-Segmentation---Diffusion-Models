import math
import os
import sys

import torch


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def load_mean_std(data_dir):
    mean_std_path = os.path.join(data_dir, "mean_std.pt")

    if os.path.exists(mean_std_path):
        mean_std = torch.load(mean_std_path)
        mean, std = mean_std["mean"], mean_std["std"]
    else:
        print("Mean and std not found. You need to calculate these with pretrain.py first.")
        sys.exit(1)

    return mean, std


def load_class_weights(data_dir):
    class_weights_path = os.path.join(data_dir, "class_weights.pt")

    if os.path.exists(class_weights_path):
        class_weights = torch.load(class_weights_path)
    else:
        print("Class weights not found. You need to calculate these with pretrain.py first.")
        sys.exit(1)

    return class_weights
