import gc

import numpy as np
import torch
import torch.nn as nn
from dataset.scripts.segmentationData import OCTSegmentationDataset
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from tqdm import tqdm

# Configuration
CHECKPOINT_PATH = "./models/combined_model/weights/Segmentation_Epoch_9.pth"
NUM_CLASSES = 6  # Including background
BATCH_SIZE = 4
SAMPLE_SIZE = 100  # Number of test samples to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_test_dataset(dataset, sample_size, random_seed=42):
    """
    Reduce the size of a dataset by random sampling.

    Args:
        dataset: The original dataset
        sample_size: Number of samples to keep
        random_seed: Random seed for reproducibility
    """
    if sample_size >= len(dataset):
        print(
            f"Warning: Requested sample size {sample_size} is >= dataset size {len(dataset)}"
        )
        return dataset

    torch.manual_seed(random_seed)
    indices = torch.randperm(len(dataset))[:sample_size].tolist()
    return Subset(dataset, indices)


def _init_model():
    """Initialize and load the segmentation model"""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # Modify classifier for our 6 classes
    model.classifier[4] = nn.Conv2d(
        model.classifier[4].in_channels, NUM_CLASSES, kernel_size=1
    )

    # Handle auxiliary classifier if present
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            model.aux_classifier[4].in_channels, NUM_CLASSES, kernel_size=1
        )

    # Load pretrained weights
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}. Using initialized model.")

    return model.to(device), DeepLabV3_ResNet50_Weights.DEFAULT.transforms()


def visualize_batch(images, masks, predictions, save_path=None):
    """Visualize a batch of images, masks and predictions"""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    # Convert tensors to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

    fig, axes = plt.subplots(BATCH_SIZE, 3, figsize=(15, 5 * BATCH_SIZE))

    for i in range(min(BATCH_SIZE, len(images))):
        # Display image
        if images[i].shape[-1] == 3:  # RGB
            axes[i, 0].imshow(images[i])
        else:  # Grayscale
            axes[i, 0].imshow(images[i, ..., 0], cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        # Create color mapping for masks
        color_map = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
        color_map[1] = [255, 0, 0]  # Red
        color_map[2] = [0, 255, 0]  # Green
        color_map[3] = [0, 0, 255]  # Blue
        color_map[4] = [255, 192, 203]  # Pink
        color_map[5] = [255, 255, 0]  # Yellow

        # Display ground truth mask with colors
        mask_rgb = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
        for j in range(NUM_CLASSES):
            mask_rgb[masks[i] == j] = color_map[j]
        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Display prediction mask with colors
        pred_rgb = np.zeros((*predictions[i].shape, 3), dtype=np.uint8)
        for j in range(NUM_CLASSES):
            pred_rgb[predictions[i] == j] = color_map[j]
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def oct_collate_fn(batch):
    """Custom collate function to handle None values."""
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    print(f"Using device: {device}")

    # Initialize model
    model, transform = _init_model()
    model.eval()

    # Initialize dataset and loader
    print("Loading dataset...")
    test_dataset = OCTSegmentationDataset(data="test", transforms=transform)

    # Reduce dataset size if needed
    reduced_dataset = reduce_test_dataset(test_dataset, sample_size=SAMPLE_SIZE)
    print(f"Using {len(reduced_dataset)} samples from {len(test_dataset)} total")

    # Create DataLoader
    test_loader = DataLoader(
        reduced_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=oct_collate_fn,
    )

    # Initialize metrics
    metrics = MetricCollection(
        {
            "Dice_Score": DiceScore(
                num_classes=NUM_CLASSES,
                average="weighted",
                input_format="index",
                include_background=False,
            ),
            "MeanIoU": MeanIoU(
                num_classes=NUM_CLASSES, input_format="index", include_background=False
            ),
            "Jaccard_Index": MulticlassJaccardIndex(
                num_classes=NUM_CLASSES, average="weighted"
            ),
        }
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=255
    )  # 255 is often used for ignore index

    # Evaluation loop
    running_loss = 0.0
    num_samples = 0
    progress_bar = tqdm(test_loader, desc="Evaluating")

    # Save a visualization of the first batch
    visualize_first_batch = True

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs["out"]

            # Calculate loss
            loss = criterion(logits, masks)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            metrics.update(preds, masks)

            # Update running loss
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Update progress bar
            progress_bar.set_postfix(loss=running_loss / num_samples)

            # Visualize first batch
            if visualize_first_batch and batch_idx == 0:
                visualize_batch(
                    images, masks, preds, save_path="batch_visualization.png"
                )
                visualize_first_batch = False

            # Clean up memory
            torch.cuda.empty_cache()
            del images, masks, outputs, logits, preds
            gc.collect()

    # Compute final metrics
    final_metrics = metrics.compute()
    avg_loss = running_loss / num_samples

    # Display results
    print("\nTest Results:")
    print(f"Loss: {avg_loss:.4f}")

    for name, value in final_metrics.items():
        if name == "MeanIoU":
            # Print per-class IoU
            print(f"{name}:")
            class_names = ["Background"] + [
                test_dataset.fluid_types.get(i, f"Class {i}")
                for i in range(1, NUM_CLASSES)
            ]
            for i, iou in enumerate(value):
                print(f"  {class_names[i]}: {iou.item():.4f}")
            print(f"  Mean: {value.mean().item():.4f}")
        else:
            print(f"{name}: {value.item():.4f}")


if __name__ == "__main__":
    main()
