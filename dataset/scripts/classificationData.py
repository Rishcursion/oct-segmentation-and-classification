import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


# Custom PyTorch Dataset
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
            for img_name in os.listdir(class_dir):
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
