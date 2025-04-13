import os
from enum import unique
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision.transforms import v2


# Custom PyTorch Dataset
class segmentationData(Dataset):
    def __init__(self, data: Literal["test", "train", "val"], transforms=None) -> None:
        if data not in ["test", "train", "val"]:
            raise ValueError("Invalid Data Directory Specified")
        self.root_dir = os.path.abspath(f"dataset/segmentation/{data}")
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
        self.class_indices = {
            (255, 0, 0): [0, "Red"],
            (0, 255, 0): [1, "Green"],
            (0, 0, 255): [2, "Blue"],
            (255, 192, 203): [3, "Pink"],
            (255, 255, 0): [4, "Yellow"],
        }
        for _, _, files in os.walk(self.root_dir):
            for file in files:
                self.image_paths.append(os.path.join(self.root_dir, file))

    def standardize_mask_pil(self, mask_pil):
        """
        Standardizes a PIL mask image using K-Means clustering.

        Args:
            mask_pil (PIL.Image): Input mask image.
            num_classes (int): Number of clusters to reduce the color variations.

        Returns:
            PIL.Image: Standardized mask with clustered labels.
        """
        # Convert PIL Image to NumPy array
        mask_np = np.array(mask_pil)  # Shape: (H, W, 3)
        # Reshape to (num_pixels, 3) for K-Means clustering
        h, w, c = mask_np.shape
        mask_flat = mask_np.reshape(-1, 3)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=6, random_state=42)
        labels = kmeans.fit_predict(
            mask_flat
        )  # Each pixel gets a cluster ID (0 to num_classes-1)

        # Reshape back to original image shape (H, W)
        clustered_mask = labels.reshape(h, w).astype(np.uint8)
        # Convert back to a PIL image
        standardized_mask = Image.fromarray(clustered_mask)

        return standardized_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int = 0):
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        local_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((520, 780), antialias=True),
                v2.ToDtype(torch.long),
            ]
        )
        image_bounds = (0, 0, 570, 380)  # Left half
        ground_truth_bounds = (570, 0, 1140, 380)  # Right half
        image = img.crop(image_bounds)
        ground_truth = img.crop(ground_truth_bounds)
        ground_truth = self.standardize_mask_pil(ground_truth)
        if self.transforms:
            image = self.transforms(image)
            ground_truth = local_transforms(ground_truth)
        return image, ground_truth
