import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


# Custom PyTorch Dataset
class segmentationData(Dataset):
    def __init__(self, data: Literal["test", "train", "val"], transforms=None) -> None:
        if data not in ["test", "train", "val"]:
            raise ValueError("Invalid Data Directory Specified")
        self.root_dir = os.path.abspath(f"./segmentation/{data}")
        self.subset = data
        self.image_paths = []
        self.transforms = transforms
        self.legend = {
            "Red": "Sub-Retinal Fluid",
            "Green": "Intra-Retinal Fluid",
            "Blue": "Pigment Epithelial Detachment",
            "Pink": "Integrity Of Inner/Outer Segment",
            "Yellow": "Subretinal hyper-reflective material",
        }
        for _, _, files in os.walk(self.root_dir):
            for file in files:
                self.image_paths.append(os.path.join(self.root_dir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int = 0):
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        image_bounds = (0, 0, 570, 380)  # Left half
        ground_truth_bounds = (570, 0, 1140, 380)  # Right half
        image = img.crop(image_bounds)
        ground_truth = img.crop(ground_truth_bounds)
        if self.transforms:
            image = self.transforms(image)
            ground_truth = self.transforms(ground_truth)
        return image, ground_truth
