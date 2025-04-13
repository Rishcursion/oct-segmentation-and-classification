import os
import random
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


class classificationData(Dataset):
    def __init__(self, root_dir: Literal["test", "train", "val"], transform=None):
        self.root_dir = os.path.join(
            os.path.abspath("./dataset/classification/"), root_dir
        )
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))  # Get class names
        self.image_paths = []
        self.labels = {}

        # Collect image paths and labels
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, cls)
            all_images = os.listdir(class_dir)
            
            # For 'normal' class, take half the images
            if cls == "NORMAL":
                # Shuffle and select half
                random.seed(42)  # For reproducibility
                random.shuffle(all_images)
                num_samples = len(all_images) // 2
                selected_images = all_images[:num_samples]
            else:
                selected_images = all_images  # Take all images for other classes
            
            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels[img_path] = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[img_path]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
