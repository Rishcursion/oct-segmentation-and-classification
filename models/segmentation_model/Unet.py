import gc
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from dataset.scripts.segmentationData import segmentationData

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_dataloader(
    subset: Literal["test", "train", "val"], transform=None, sample_size=16
):
    dataset = segmentationData(subset, transforms=transform)
    indices = np.random.choice(
        len(dataset), size=sample_size, replace=False
    )  # Select random samples
    sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset=dataset, batch_size=4, sampler=sampler)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


def _init_model() -> tuple[nn.Module, transforms.Compose]:
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, 5, kernel_size=1)
    if getattr(model, "aux_classifier"):
        model.aux_classifier[4] = nn.Conv2d(
            model.aux_classifier[4].in_channels, 5, kernel_size=1
        )
        torch.nn.init.xavier_normal_(model.aux_classifier[4].weight)
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.classifier[4].parameters():
        param.requires_grad = True
    torch.nn.init.xavier_normal_(model.classifier[4].weight)
    return model, DeepLabV3_ResNet50_Weights.DEFAULT.transforms()


def train(
    model: nn.Module,
    optimizer: optim.AdamW,
    criterion: nn.Module,
    data,
    num_iters: int = 5,
):
    model.cuda()
    start_time = time.time()
    results = {"train": {}, "val": {}}
    metrics = MetricCollection(
        {
            "Dice Score": DiceScore(num_classes=5, average="macro",include_background=False,input_format="index"),
            "IoU": MeanIoU(num_classes=5,include_background=False,input_format="index"),
        }
    ).to(device=device)
    best_dice_score = -1
    for epoch in range(num_iters):
        print(f"\nEpoch {epoch+1}/{num_iters}\n" + "-" * 10)
        epoch_start_time = time.time()
        metrics.reset()
        for phase in ["train", "val"]:
            phase_start_time = time.time()
            running_loss = 0.0
            print(f"Starting {phase.capitalize()} Phase: \n")
            model.train(phase == "train")
            for batch_id, (img, mask) in enumerate(data[phase]):
                print(
                    f"\rProgress: {(batch_id + 1)} / {len(data[phase])}",
                    end="",
                    flush=True,
                )
                optimizer.zero_grad()
                img, mask = img.to(device), mask.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    pred_mask = model(img)
                    main_loss = criterion(pred_mask["out"], mask)
                    aux_loss = criterion(pred_mask["aux"], mask)
                    loss = main_loss + 0.4 * aux_loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                pred_mask = torch.argmax(pred_mask["out"], dim=1)
                running_loss += loss.item() * img.size(0)
                metrics.update(pred_mask, mask)
                torch.cuda.empty_cache()
                del pred_mask, mask, img
                gc.collect()
            avg_loss = running_loss / len(data[phase].dataset)
            results[phase][epoch] = metrics.compute()
            results[phase][epoch]["loss"] = avg_loss
            if phase == "val" and results[phase][epoch]["Dice Score"] > best_dice_score:
                best_dice_score = results[phase][epoch]["Dice Score"]
                torch.save(
                    model.state_dict(),
                    f"models/segmentation_model/Segmentation Epoch {epoch}.json",
                )
            print(
                f"""{phase.capitalize()} Phase Complete
                Time Taken: {(time.time() - phase_start_time)/60:.2f}
                Dice Score: {results[phase][epoch]['Dice Score']}
                Mean IoU: {results[phase][epoch]['IoU']}
                Loss: {results[phase][epoch]['loss']}"""
            )
        print(
            f"""Epoch Complete
        Time Taken: {(time.time()- epoch_start_time)/60:.2f}
        Best Dice Score: {best_dice_score}"""
        )
    print(
        f"Model Training And Validation Completed In: {time.time()/60- start_time/60:.2f} Minutes"
    )
    return results


if __name__ == "__main__":
    epochs = int(input("Enter Number Of Epochs To Train The Model For: "))
    model, transform = _init_model()
    data = {
        "train": _init_dataloader("train", transform),
        "val": _init_dataloader("val", transform),
    }
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-6)
    criterion = nn.CrossEntropyLoss()
    history = train(model, optimizer, criterion, data, epochs)
    with open("models/saved_models_metrics/segmentation_metrics.json", "w") as fp:
        import json

        json.dump(history, fp,default=float)
