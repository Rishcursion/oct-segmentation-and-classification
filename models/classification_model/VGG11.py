import gc
import time
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchmetrics import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)
from torchvision.models import VGG11_BN_Weights, vgg11_bn

from dataset.scripts.classificationData import classificationData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_summary(device=device, abbreviated=False)


def weights_init_uniform_rule(m):
    """Custom weight initialization for linear layers."""
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def load_model(model: nn.Module) -> nn.Module:
    model.load_state_dict(
        torch.load("./models/classification_model/first_run/classification_model.pth")
    )
    return model


def _init_model() -> tuple[nn.Module, nn.Module]:
    """Initializes the VGG11_BN model with modified classifier for 4-class classification."""
    model_weights = VGG11_BN_Weights.DEFAULT
    transforms = model_weights.transforms()

    model = vgg11_bn(weights=model_weights)

    # Freeze all feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 4)])  # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    # Apply custom weight initialization to classifier layers
    model.classifier.apply(weights_init_uniform_rule)
    print(model.classifier)
    return transforms, model.to(device)


def _init_dataloader(
    subset: Literal["train", "val"], batches: int = 8, transforms=None,return_weights= True
):
    dataset = classificationData(subset, transform=transforms)
    class_counts = Counter(dataset.labels.values())
    class_weights = {label: sum(class_counts.values())/(4*count) for label,count in class_counts.items()}
    if return_weights:
        return DataLoader(
            dataset,
            batch_size=batches,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        ),class_weights
    else:
        return DataLoader(
            dataset,
            batch_size=batches,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


def train_model(vgg, criterion, optimizer, dataloaders, num_epochs=10):
    since = time.time()
    vgg.cuda()
    best_acc = 0.0

    # Initialize the GradScaler for mixed precision
    # Initialize Metrics
    metrics = MetricCollection(
        {
            "ConfusionMatrix": ConfusionMatrix(task="multiclass", num_classes=4),
            "Accuracy": Accuracy(task="multiclass", num_classes=4, average="weighted"),
            "F1Score": F1Score(task="multiclass", num_classes=4, average="weighted"),
            "AUROC": AUROC(task="multiclass", num_classes=4, average="weighted"),
            "Recall": Recall(task="multiclass", num_classes=4, average="weighted"),
            "Precision": Precision(
                task="multiclass", num_classes=4, average="weighted"
            ),
        }
    ).to(device)

    print(f"\n{'='*40}\nStarting Model Training And Validation\n{'='*40}\n")
    train_results = defaultdict(dict)
    val_results = defaultdict(dict)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n" + "-" * 10)

        # Training Phase
        vgg.train()
        train_loss, train_correct = 0.0, 0
        total_train_samples = 0
        metrics.reset()
        train_start_time = time.time()
        for i, (inputs, labels) in enumerate(dataloaders["TRAIN"][0]):
            print(
                f"\rProgress: {(i + 1)} / {len(dataloaders['TRAIN'][0])}",
                end="",
                flush=True,
            )
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step() 

            preds = outputs.argmax(dim=1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)
            metrics.update(outputs, labels)

            # Clear variables to free memory
            del inputs, labels, preds, outputs
            gc.collect()
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / total_train_samples
        avg_train_acc = train_correct / total_train_samples
        train_results[epoch] = metrics.compute()

        # Validation Phase
        vgg.eval()
        val_loss, val_correct = 0.0, 0
        total_val_samples = 0
        metrics.reset()
        val_start_time = time.time()
        print("\nTraining Phase Complete\n")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}\nTime Taken: {(time.time()-train_start_time)/60:.2f} Minutes\n"
        )
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders["VAL"]):
                print(
                    f"\rProgress: {(i + 1)} / {len(dataloaders['VAL'])}",
                    end="",
                    flush=True,
                )
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = vgg(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)
                metrics.update(outputs, labels)

                del inputs, labels, preds, outputs
                gc.collect()
                torch.cuda.empty_cache()
        avg_val_loss = val_loss / total_val_samples
        avg_val_acc = val_correct / total_val_samples
        val_results[epoch] = metrics.compute()
        print("\nValidation Phase Complete\n")
        print(
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}\nTime Taken: {(time.time()-val_start_time)/60:.2f} Minutes\n"
        )

        # Save best model
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(
                vgg.state_dict(),
                f"./models/classification_model/classification_model_{epoch}.pth",
            )

    elapsed_time = time.time() - since
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return train_results, val_results


def _convert_metrics(metrics_dict):
    converted = {}
    for epoch, metrics in metrics_dict.items():
        converted[epoch] = {
            key: value.tolist() if torch.is_tensor(value) else value
            for key, value in metrics.items()
        }
    return converted


if __name__ == "__main__":
    transforms, model = _init_model()
    # model = load_model(model)
    dataloader = {
        "VAL": _init_dataloader("val", transforms=transforms,return_weights=False),
        "TRAIN": _init_dataloader("train", transforms=transforms),
    }
    model.cuda()
    optimizer = optim.AdamW(lr=1e-4, weight_decay=3e-6, params=model.parameters())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(dataloader["TRAIN"][1].values()),dtype=torch.float32,device=device))
    train_metrics, val_metrics = train_model(
        model, criterion, optimizer, dataloader, num_epochs=6   
    )
    import json

    with open(
        "./models/classification_model/saved_models_metrics/classification/train_metrics_2.json",
        "w",
    ) as fp:
        json.dump(_convert_metrics(train_metrics), fp)
    with open(
        "./models/classification_model/saved_models_metrics/classification/val_metrics_2.json",
        "w",
    ) as fp:
        json.dump(_convert_metrics(val_metrics), fp)
