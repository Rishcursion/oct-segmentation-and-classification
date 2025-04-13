import gc
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from dataset.scripts.segmentationData import segmentationData
from torch.utils.data import DataLoader, dataset
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation import DiceScore,MeanIoU
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_dataloader(subset: Literal["test", "train", "val"], transform=None):
    dataset = segmentationData(subset, transforms=transform)
    
    return DataLoader(
        dataset=dataset ,shuffle=True, batch_size=4
    )


def _init_model() -> tuple[nn.Module, torch.nn.Module]:
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, 6, kernel_size=1)
    for param in model.classifier.parameters():
        param.requires_grad = False
    if hasattr(model, "aux_classifier"):
        model.aux_classifier[4] = nn.Conv2d(
            model.aux_classifier[4].in_channels, 6, kernel_size=1
        )
        nn.init.xavier_normal_(model.aux_classifier[4].weight)
    for param in model.classifier[4].parameters():
        param.requires_grad = True
    nn.init.xavier_normal_(model.classifier[4].weight)
    return model.to(device), DeepLabV3_ResNet50_Weights.DEFAULT.transforms()


def train(model, optimizer, criterion, data, num_iters=5):
    model.train()
    scaler = torch.GradScaler()  # Mixed Precision Scaler
    metrics = MetricCollection(
        {
            "Dice_Score": DiceScore(
                num_classes=6, average="weighted", input_format="index"
            ),
            "MeanIoU":MeanIoU(num_classes=6, input_format="index"),
            "Jaccard Index": MulticlassJaccardIndex(num_classes=6, average="weighted")
        }
    ).to(device)
    best_dice_score = -1
    results = {}
    start_time = time.time()
    print(f"""
    {'='*40}
    Starting Model Training
    Number Of Epochs: {num_iters}
    Model Name: deeplabv3_resnet50
    Metrics Used: Dice Score, Intersection Over Union
    {'='*40}
    """)
    for epoch in range(num_iters):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_iters}\n" + "-" * 10)
        metrics.reset()
        avg_loss = 0
        for phase in ["train", "val"]:
            phase_start_time = time.time()
            running_loss = 0.0
            model.train(phase == "train")
            print(f"Starting  {phase.capitalize()} Phase")
            for batch_id, (img, mask) in enumerate(data[phase]):
                print(f"\rBatch: {batch_id+1}/ {len(data[phase])} | Loss: {running_loss/len(data[phase].dataset):.2f}", end="", flush=True)
                img, mask = img.to(device), mask.to(device).squeeze(1)
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda"):  # Mixed Precision Forward Pass
                    pred_mask = model(img)
                    main_loss = criterion(pred_mask["out"].squeeze(1), mask)
                    aux_loss = criterion(pred_mask.get("aux", 0), mask)
                    loss = main_loss + 0.4 * aux_loss

                if phase == "train":
                    scaler.scale(loss).backward()  # Scaled Backward Pass
                    scaler.step(optimizer)
                    scaler.update()

                pred_mask = torch.argmax(pred_mask["out"].squeeze(1), dim=1)
                running_loss += loss.item() * img.size(0)
                metrics.update(pred_mask, mask)

                torch.cuda.empty_cache()
                del pred_mask, mask, img
                gc.collect()

            avg_loss = running_loss / len(data[phase].dataset)
            results[epoch] = metrics.compute()
            results[epoch]["Loss"] = avg_loss

            if phase == "val" and results[epoch]["Dice_Score"] > best_dice_score:
                best_dice_score = results[epoch]["Dice_Score"]
                torch.save(
                    model.state_dict(),
                    f"models/segmentation_model/saved_models/Segmentation_Epoch_{epoch}.pth",
                )

            print(
                f"""\n{phase.capitalize()} Completed
                    Dice: {results[epoch]['Dice_Score']:.4f}
                    Jaccard Index: {results[epoch]['Jaccard Index']:.4f}
                    MeanIoU: {results[epoch]["MeanIoU"]:.4f}
                    Loss: {results[epoch]['Loss']:.4f}
                    Time Taken: {(time.time() - phase_start_time)/60:.2f} Minutes"""
            )
        print(
            f"""Epoch Completed
        Time Taken: {(time.time() - epoch_start_time)/60:.2f} Minutes
        Best Dice Score: {best_dice_score}"""
        )
    print(
        f"""Model Training Completed
        Time Taken:  {(time.time()-start_time)/60:.2f} Minutes
        Best Dice Score: {best_dice_score}"""
    )
    return results


if __name__ == "__main__":
    epochs = int(input("Enter Number Of Epochs: "))
    model, transform = _init_model()
    data = {
        "train": _init_dataloader("train", transform),
        "val": _init_dataloader("val", transform),
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-6)
    criterion = nn.CrossEntropyLoss() 
    history = train(model, optimizer, criterion, data, epochs)
    with open("models/segmentation_model/metrics/TrainResults.json", "w") as fp:
        import json

        json.dump(history, fp)
