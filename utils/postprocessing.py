# Define color palette for the segmentation classes
import torch

COLORS = torch.tensor(
    [
        [0, 0, 0],  # Background (Black)
        [255, 0, 0],  # Class 1 (Red)
        [0, 255, 0],  # Class 2 (Green)
        [0, 0, 255],  # Class 3 (Blue)
        [255, 255, 0],  # Class 4 (Yellow)
        [255, 0, 255],  # Class 5 (Magenta)
    ],
    dtype=torch.uint8,
)


def apply_argmax_and_coloring(sampled_images):
    # Apply argmax to get the segmentation class for each pixel
    segmentation = torch.argmax(sampled_images, dim=1)
    device = sampled_images.device

    # Create a blank RGB image
    colored_image = torch.zeros(
        (sampled_images.size(0), 3, sampled_images.size(2), sampled_images.size(3)),
        dtype=torch.uint8,
        device=device,
    )

    COLORS_ = COLORS.to(device)

    # Apply colors to the segmentation classes
    for class_idx in range(COLORS.size(0)):
        mask = segmentation == class_idx
        for color_idx in range(3):
            colored_image[:, color_idx, :, :] += mask * COLORS_[class_idx, color_idx]

    return colored_image
