from utils import plot_confusion_matrix
from utils.preprocessing import load_mri
import torch

nifti = load_mri("data/splitted/test_masks/9250756.nii.gz")
img = torch.from_numpy(nifti[0])

nifti = load_mri("results/examples/example_9250756.nii.gz")
pred = torch.from_numpy(nifti[0])

plot_confusion_matrix(img.flatten(), pred.flatten(), ["Pozadina","Femur","Hrskavica femura","Tibia","Med. hrskavica tibie","Lat. hrskavica tibie"], normalize=True, title='Confusion Matrix')