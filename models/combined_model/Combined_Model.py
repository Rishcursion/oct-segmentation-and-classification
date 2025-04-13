import enum
import gc
import json
import os
from pathlib import Path
import sched
from typing import Any, Dict, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import GradScaler, autocast
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import DataLoader,Subset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
import random
from torchvision.models import VGG11_BN_Weights, vgg11_bn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50

# Assuming these are custom dataset classes
from dataset.scripts.classificationData import classificationData
from dataset.scripts.segmentationData import segmentationData

# Configure environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create directory for model checkpoints if it doesn't exist
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Configure device and memory settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Optimize for speed with fixed-size inputs
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Initial cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get device information
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"Available VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Set optimal memory fraction based on available VRAM
    if gpu_props.total_memory < 6 * 1024**3:  # Less than 6GB VRAM
        print("Low VRAM detected, applying strict memory optimizations")
        # Lower batch sizes will be used
else:
    print("CUDA not available, using CPU")


class FluidClass(enum.Enum):
    """Enumeration for fluid classes in OCT segmentation."""
    BACKGROUND = 0
    RED = 1       # First fluid type
    GREEN = 2     # Second fluid type
    BLUE = 3      # Third fluid type
    PINK = 4      # Fourth fluid type
    YELLOW = 5    # Fifth fluid type


class RetinalCondition(enum.Enum):
    """Enumeration for retinal conditions in classification."""
    NORMAL = 0
    DRUSEN = 1
    DME = 2
    CNV = 3


def init_segmentation_model() -> Tuple[nn.Module, Any]:
    """Initialize segmentation model with memory-efficient configuration.
    
    Returns:
        Tuple containing the model and its associated transforms
    """
    # Free memory before loading model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load DeepLabV3 with ResNet50 backbone
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    
    # Freeze backbone parameters to save memory and computation
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Modify classifier to have 6 output channels (5 fluid classes + background)
    original_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(original_channels, 6, kernel_size=1)
    
    # Initialize weights with Xavier normal for better convergence
    nn.init.xavier_normal_(model.classifier[4].weight)
    
    # Freeze all classifier parameters except the final layer
    for i, param in enumerate(model.classifier.parameters()):
        if i < len(list(model.classifier.parameters())) - 2:  # Keep last layer trainable
            param.requires_grad = False
    
    # Handle auxiliary classifier if present
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        aux_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_channels, 6, kernel_size=1)
        nn.init.xavier_normal_(model.aux_classifier[4].weight)
        # Freeze auxiliary classifier
        for param in model.aux_classifier.parameters():
            param.requires_grad = False
    
    # Try to load pre-trained weights with proper error handling
    weights_path = Path("models/combined_model/weights/Segmentation_Epoch_9.pth")
    if weights_path.exists():
        try:
            # Load with map_location to avoid OOM
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded segmentation weights from {weights_path}")
        except Exception as e:
            print(f"Error loading segmentation model weights: {e}")
    else:
        print(f"Warning: Segmentation weights not found at {weights_path}")
    
    # Enable gradient checkpointing to save memory during backprop
    if hasattr(model.backbone, "layer4"):
        model.backbone.layer4.apply(lambda m: setattr(m, "checkpoint", True))
    
    # Set model to evaluation mode
    model.eval()
    
    return model.to(device), DeepLabV3_ResNet50_Weights.DEFAULT.transforms()


class EnhancedVGG(nn.Module):
    """Enhanced VGG model with custom input layer for OCT image classification."""
    
    def __init__(self, num_classes: int = 4, input_channels: int = 9):
        """Initialize enhanced VGG model.
        
        Args:
            num_classes: Number of classification categories
            input_channels: Number of input channels (RGB + segmentation masks)
        """
        super().__init__()
        # Start with pre-trained VGG11 with batch normalization
        self.model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        
        # Replace first convolutional layer to handle additional input channels
        original_conv = self.model.features[0]
        self.model.features[0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )
        
        # Initialize first layer with Kaiming normal for ReLU
        init.kaiming_normal_(
            self.model.features[0].weight, mode="fan_out", nonlinearity="relu"
        )
        
        # Modify final classification layer for retinal conditions
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.model.classifier[-2] = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


def init_models() -> Tuple[nn.Module, nn.Module, Any]:
    """Initialize both segmentation and classification models with improved architecture.
    
    Returns:
        Tuple containing segmentor, classifier, and transforms
    """
    # Free memory before initialization
    gc.collect()
    torch.cuda.empty_cache()
    
    # Initialize segmentation model
    segmentor, seg_transforms = init_segmentation_model()
    segmentor.eval()  # Keep in eval mode
    
    # Initialize enhanced VGG classifier
    classifier = EnhancedVGG(num_classes=4, input_channels=9)
    
    # Return initialized models and transforms
    return segmentor, classifier, VGG11_BN_Weights.DEFAULT.transforms()


def init_dataloaders(
    task: Literal["segmentation", "classification"],
    option: Literal["test", "train", "val"],
    transforms: Any,
    batch_size: Optional[int] = 32,
) -> DataLoader:
    """Initialize memory-optimized dataloader.
    
    Args:
        task: Type of dataset to load
        option: Dataset split to use
        transforms: Transforms to apply to the data
        batch_size: Optional custom batch size override
        
    Returns:
        Configured DataLoader
    """
    print(f"Inititalizing {option} dataset...")
    # Determine batch size based on available memory
    if batch_size is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            batch_size = 2 if vram_gb < 6 else (4 if vram_gb < 10 else 8)
        else:
            batch_size = 4  # Default for CPU
    
    # Choose appropriate dataset
    if task == "classification":
        dataset = classificationData(option, transforms)
        normal_idx = []
        abnormal_idx = []
        for i, (_, label) in enumerate(dataset):
            if label==3:
                normal_idx.append(i)
            else:
                abnormal_idx.append(i)
        selected_normal = random.sample(normal_idx, int(len(normal_idx)//3))
        final_idx = abnormal_idx+selected_normal
        dataset = Subset(dataset,final_idx)
    else:
        dataset = segmentationData(option, transforms)
    print("DONE!!")
    # Configure dataloader with memory optimizations
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),  # Limit workers for lower memory
        shuffle=(option == "train"),  # Only shuffle training data
        pin_memory=torch.cuda.is_available(),  # Pin memory if using CUDA
        persistent_workers=True,  # Keep workers alive between batches
        drop_last=False,  # Don't drop incomplete last batch
    )


class ModelAggregator(nn.Module):
    """Combined model that uses segmentation outputs to improve classification."""
    
    def __init__(self) -> None:
        """Initialize the aggregated model."""
        super().__init__()
        # Load models and transforms
        self.segmentor, self.classifier, self.transforms = init_models()
        
        # Freeze segmentation model to prevent updating
        for param in self.segmentor.parameters():
            param.requires_grad = False
        self.segmentor.eval()
        
        # Store segmentation transforms
        self.seg_transforms = DeepLabV3_ResNet50_Weights.DEFAULT.transforms()
        
        # Set up comprehensive metrics collection for the 4 classes
        self.metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=4, average="macro"),
                "f1_score": MulticlassF1Score(num_classes=4, average="macro"),
                "precision": MulticlassPrecision(num_classes=4, average="macro"),
                "recall": MulticlassRecall(num_classes=4, average="macro"),
                "auroc": MulticlassAUROC(num_classes=4),
            }
        )
        
        # Debug flag for development
        self.debug_mode = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through segmentation and classification pipeline.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Classification logits [B, 4]
        """
        # Optional debugging output
        if self.debug_mode:
            print(f"DEBUG: Input shape: {x.shape}")
            
        # Generate segmentation masks with mixed precision
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
            # Apply transforms expected by segmentation model
            seg_input = self.seg_transforms(x)
            
            # Get segmentation output
            masks = self.segmentor(seg_input)["out"]
            
            if self.debug_mode:
                print(f"DEBUG: Segmentation output shape: {masks.shape}")
                print(f"DEBUG: Number of output channels: {masks.shape[1]}")
            
            # Resize masks to match input image dimensions
            if masks.shape[2:] != x.shape[2:]:
                masks = F.interpolate(
                    masks, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            
            # Apply softmax to get probability maps
            masks = F.softmax(masks, dim=1)
            
            if self.debug_mode:
                print(f"DEBUG: Resized masks shape: {masks.shape}")
                has_nan = torch.isnan(masks).any()
                print(f"DEBUG: Contains NaN values: {has_nan}")
        
        # Prepare classification input
        cls_input = self.transforms(x)
        
        # Combine original input with segmentation masks
        combined = torch.cat([cls_input, masks], dim=1)
        
        if self.debug_mode:
            print(f"DEBUG: Combined input shape: {combined.shape}")
        
        # Run classification with mixed precision
        with autocast(device_type="cuda", dtype=torch.float16):
            output = self.classifier(combined)
        
        if self.debug_mode:
            print(f"DEBUG: Classification output shape: {output.shape}")
        
        return output


def train_model(
    model: ModelAggregator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: Union[str, torch.device] = "cuda",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[ModelAggregator, Dict]:
    """Train the model with enhanced monitoring and memory management.
    
    Args:
        model: The combined model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        num_epochs: Number of training epochs
        device: Device to train on
        scheduler: Optional learning rate scheduler
        
    Returns:
        Tuple of trained model and training metrics
    """
    # Initialize results dictionary to store metrics
    results = {
        "train": {
            "accuracy": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "auroc": [],
            "loss": [],
        },
        "val": {
            "accuracy": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "auroc": [],
            "loss": [],
        },
    }
    
    # Move model to device
    model.cuda()
    
    
    # Initialize train and validation metrics
    train_metrics = model.metrics.clone().to(device)
    val_metrics = model.metrics.clone().to(device)
    
    # Initialize gradient scaler for mixed precision training    
    # Track best validation metrics for model saving
    best_val_accuracy = 0.0
    
    # Reset memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Training loop
    for epoch in range(num_epochs-3, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.classifier.cuda().train()
        model.segmentor.cuda().eval()  # Keep segmentation model in eval mode
        running_loss = 0.0
        
        # Reset metrics for new epoch
        train_metrics.reset()
        
        # Clear memory before epoch
        gc.collect()
        torch.cuda.empty_cache()
        
        # Training batch loop
        batch_count = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            # Progress indicator
            print(
                    f"\rBatch: {i+1}/{batch_count} | Loss: {running_loss/max(1, (i+1)):.6f}",
                    flush=True,end=""
                )
            
            # Move data to device with non_blocking for async transfer
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients (set_to_none is more memory efficient)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling for mixed precision
            loss.backward()
            
            # Apply gradient clipping to prevent explosion
            if hasattr(optimizer, "clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            optimizer.step()            
            # Update running loss (use item() to detach from graph)
            running_loss += loss.item()
            
            # Update metrics
            with torch.no_grad():
                train_metrics.update(outputs.float(), labels)
            
            # Periodically move model to CPU to consolidate memory
            if i % 20 == 0 and i > 0:
                # Save current state
                model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                model.to(device)
                # Restore state
                model.load_state_dict(model_state)
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step(running_loss/len(train_loader))
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_metrics = train_metrics.compute()
        
        # Store training metrics
        results["train"]["loss"].append(epoch_loss)
        results["train"]["accuracy"].append(epoch_metrics["accuracy"].item())
        results["train"]["f1_score"].append(epoch_metrics["f1_score"].item())
        results["train"]["precision"].append(epoch_metrics["precision"].item())
        results["train"]["recall"].append(epoch_metrics["recall"].item())
        results["train"]["auroc"].append(epoch_metrics["auroc"].item())
        
        # Print training metrics
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Train Metrics:")
        print(f"  Accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {epoch_metrics['f1_score']:.4f}")
        print(f"  Precision: {epoch_metrics['precision']:.4f}")
        print(f"  Recall: {epoch_metrics['recall']:.4f}")
        print(f"  AUROC: {epoch_metrics['auroc']:.4f}")
        
        # Clear memory before validation
        gc.collect()
        torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        # Reset metrics for validation
        val_metrics.reset()
        
        # Validation batch loop
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                # Progress indicator
                if i % max(1, len(val_loader) // 5) == 0:
                    print(
                        f"Val Batch: {i+1}/{len(val_loader)} | "
                        f"Loss: {running_loss/max(1, (i+1)):.6f}",
                        flush=True
                    )
                
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update running loss
                running_loss += loss.item()
                
                # Update metrics
                val_metrics.update(outputs.float(), labels)
        
        # Calculate validation metrics
        val_loss = running_loss / len(val_loader)
        epoch_val_metrics = val_metrics.compute()
        
        # Store validation metrics
        results["val"]["loss"].append(val_loss)
        results["val"]["accuracy"].append(epoch_val_metrics["accuracy"].item())
        results["val"]["f1_score"].append(epoch_val_metrics["f1_score"].item())
        results["val"]["precision"].append(epoch_val_metrics["precision"].item())
        results["val"]["recall"].append(epoch_val_metrics["recall"].item())
        results["val"]["auroc"].append(epoch_val_metrics["auroc"].item())
        
        # Print validation metrics
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        print(f"  Accuracy: {epoch_val_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {epoch_val_metrics['f1_score']:.4f}")
        print(f"  Precision: {epoch_val_metrics['precision']:.4f}")
        print(f"  Recall: {epoch_val_metrics['recall']:.4f}")
        print(f"  AUROC: {epoch_val_metrics['auroc']:.4f}")
        
        # Report memory usage
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Max GPU memory used: {max_memory:.2f} MB")
            torch.cuda.reset_peak_memory_stats()
        
        # Save metrics to JSON file after each epoch
        save_metrics_to_json(
            results, 
            CHECKPOINT_DIR / f"training_metrics_epoch_{epoch+1}.json"
        )
        
        # Save model checkpoint to CPU to avoid OOM during save
        checkpoint_path = CHECKPOINT_DIR / f"model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            },
            checkpoint_path,
        )
        
        # Save best model based on validation accuracy
        current_accuracy = epoch_val_metrics["accuracy"].item()
        if current_accuracy > best_val_accuracy:
            best_val_accuracy = current_accuracy
            best_model_path = CHECKPOINT_DIR / f"best_model_epoch_{epoch+1}.pth"
            
            # Save to CPU to avoid OOM
            model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model_cpu_state, best_model_path)
            print(f"Saved new best model at epoch {epoch+1} with accuracy {best_val_accuracy:.4f}")
        
        # Clear memory after epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save final results
    save_metrics_to_json(results, CHECKPOINT_DIR / "final_training_metrics.json")
    
    return model, results


def save_metrics_to_json(metrics_dict: Dict, filename: Union[str, Path]) -> None:
    """Save metrics dictionary to a JSON file with proper formatting.
    
    Args:
        metrics_dict: Dictionary of metrics to save
        filename: Path to save the JSON file
    """
    # Ensure path exists
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any tensor values to Python native types
    processed_dict = {}
    for split, metrics in metrics_dict.items():
        processed_dict[split] = {}
        for metric_name, values in metrics.items():
            # Convert tensor values to Python floats if needed
            processed_dict[split][metric_name] = [
                float(val) if hasattr(val, "item") else val for val in values
            ]
    
    # Write to JSON file with indentation for readability
    with open(filename, "w") as f:
        json.dump(processed_dict, f, indent=4)
    
    print(f"Metrics saved to {filename}")


def test_model(
    model: ModelAggregator, 
    test_loader: DataLoader, 
    device: Union[str, torch.device] = "cuda",
    save_predictions: bool = False,
) -> Dict:
    """Test the model on the test set and report metrics.
    
    Args:
        model: Model to test
        test_loader: DataLoader for test data
        device: Device to test on
        save_predictions: Whether to save model predictions
        
    Returns:
        Dictionary of test metrics
    """
    # Clear memory before testing
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Initialize test metrics
    test_metrics = model.metrics.clone().to(device)
    test_metrics.reset()
    
    # Storage for predictions if requested
    all_predictions = []
    all_labels = []
    
    # Test loop
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # Progress indicator
            if batch_idx % max(1, len(test_loader) // 5) == 0:
                print(f"Testing batch {batch_idx+1}/{len(test_loader)}")
                
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
            
            # Update metrics
            test_metrics.update(outputs.float(), labels)
            
            # Store predictions if requested
            if save_predictions:
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
    
    # Calculate and return test metrics
    computed_metrics = test_metrics.compute()
    
    # Convert metrics to regular Python types for JSON serialization
    results = {
        name: value.item() if hasattr(value, "item") else float(value)
        for name, value in computed_metrics.items()
    }
    
    # Print metrics
    print("Test Metrics:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1_score']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  AUROC: {results['auroc']:.4f}")
    
    # Save predictions if requested
    if save_predictions:
        prediction_data = {
            "predictions": all_predictions,
            "ground_truth": all_labels,
            "metrics": results,
            "classes": [c.name for c in RetinalCondition]
        }
        
        # Save predictions to file
        prediction_path = CHECKPOINT_DIR / "test_predictions.json"
        with open(prediction_path, "w") as f:
            json.dump(prediction_data, f, indent=4)
        print(f"Predictions saved to {prediction_path}")
    
    # Save metrics to JSON
    metrics_path = CHECKPOINT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Test metrics saved to {metrics_path}")
    
    return results


def visualize_metrics(results: Dict) -> None:
    """Visualize training and validation metrics.
    
    Args:
        results: Dictionary of training results
    """
    try:
        import matplotlib.pyplot as plt
        
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure and grid
        epochs = range(1, len(results["train"]["accuracy"]) + 1)
        fig = plt.figure(figsize=(18, 12))
        
        # Configure subplots in a 2x3 grid
        metric_plots = [
            ("accuracy", "Accuracy", 1),
            ("f1_score", "F1 Score", 2),
            ("precision", "Precision", 3),
            ("recall", "Recall", 4),
            ("auroc", "AUROC", 5),
            ("loss", "Loss", 6)
        ]
        
        for metric_name, title, position in metric_plots:
            ax = fig.add_subplot(2, 3, position)
            
            # Plot training and validation metrics
            train_values = results["train"][metric_name]
            val_values = results["val"][metric_name]
            
            ax.plot(epochs, train_values, 'b-', linewidth=2, label=f"Training {title}")
            ax.plot(epochs, val_values, 'r-', linewidth=2, label=f"Validation {title}")
            
            # Add horizontal line at best validation score for non-loss metrics
            if metric_name != "loss":
                best_val = max(val_values)
                best_epoch = val_values.index(best_val) + 1
                ax.axhline(y=best_val, color='g', linestyle='--', alpha=0.7,
                           label=f"Best: {best_val:.4f} (Epoch {best_epoch})")
            else:
                # For loss, we want the minimum
                best_val = min(val_values)
                best_epoch = val_values.index(best_val) + 1
                ax.axhline(y=best_val, color='g', linestyle='--', alpha=0.7,
                           label=f"Best: {best_val:.4f} (Epoch {best_epoch})")
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle('Training and Validation Metrics', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
       # Save figure
        metrics_plot_path = CHECKPOINT_DIR / "training_metrics_plot.png"
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics visualization saved to {metrics_plot_path}")
        
        # Create learning curve comparison plot
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Plot accuracy vs loss
        epochs_array = np.array(epochs)
        ax.plot(results["train"]["loss"], results["train"]["accuracy"], 'bo-', label="Training")
        ax.plot(results["val"]["loss"], results["val"]["accuracy"], 'ro-', label="Validation")
        
        # Add epoch annotations
        for i, (x, y) in enumerate(zip(results["val"]["loss"], results["val"]["accuracy"])):
            ax.annotate(f"{i+1}", (x, y), fontsize=9)
        
        ax.set_xlabel('Loss', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Loss Learning Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save learning curve plot
        learning_curve_path = CHECKPOINT_DIR / "learning_curve.png"
        plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to {learning_curve_path}")
    
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")


def load_best_model(checkpoint_dir: Path = CHECKPOINT_DIR) -> ModelAggregator:
    """Load the best model checkpoint from the specified directory.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        Best trained model
    """
    # Initialize a new model
    model = ModelAggregator()
    
    # Find best model checkpoint
    model_files = list(checkpoint_dir.glob("best_model_*.pth"))
    if not model_files:
        print("No best model checkpoint found, checking for latest checkpoint")
        model_files = list(checkpoint_dir.glob("model_checkpoint_*.pth"))
    
    if not model_files:
        print("No model checkpoints found")
        return model
    
    # Sort by epoch number
    model_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    best_model_path = model_files[-1]  # Get latest
    
    try:
        # Load model state dict
        if best_model_path.name.startswith("best_model"):
            # Best model files contain only the state dict
            state_dict = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            # Full checkpoints contain more data
            checkpoint = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Loaded model from {best_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return model




def main():
    """Main function to run training and testing."""
    # Import numpy if visualization is needed
    try:
        import numpy as np
    except ImportError:
        pass
    
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description="OCT Segmentation and Classification")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "inference"],
                       help="Mode to run in: train, test, or inference")
    parser.add_argument("--model", type=str, help="Path to model weights for test/inference")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for dataloaders")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Run in specified mode
    if args.mode == "train":
        print("Initializing training...")
        
        # Create model
        model = ModelAggregator()
        model.debug_mode = args.debug
        checkpoint = torch.load("checkpoints/model_checkpoint_epoch_3.pth", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.cuda()
        # Create dataloaders
        train_loader = init_dataloaders("classification", "train", model.transforms, 16)
        val_loader = init_dataloaders("classification", "val", model.transforms, 16)
        
        # Create loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), 
            lr=args.lr, 
            weight_decay=3e-4
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Create learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Using device: {device}")
        print(f"Training for {args.epochs} epochs...")
        
        # Train the model
        trained_model, results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs,
            device=device,
            scheduler=scheduler
        )
        
        # Visualize training metrics
        visualize_metrics(results)
        
        # Test model after training
        print("Testing trained model...")
        test_loader = init_dataloaders("classification", "test", model.transforms, args.batch_size)
        test_metrics = test_model(
            model=trained_model, 
            test_loader=test_loader, 
            device=device,
            save_predictions=True
        )
        
    elif args.mode == "test":
        print("Running model testing...")
        
        if args.model:
            # Load specific model
            model = ModelAggregator()
            model_path = Path(args.model)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            # Load best model
            model = load_best_model()
            
        # Test the model
        test_loader = init_dataloaders("classification", "test", model.transforms, args.batch_size)
        test_metrics = test_model(
            model=model, 
            test_loader=test_loader, 
            device=device,
            save_predictions=True
        )
            
    elif args.mode == "inference":
        print("Running inference...")
        
        if not args.model or not args.image:
            print("Error: Both --model and --image arguments are required for inference mode")
            return
            
        # Run inference
        result = load_and_run_inference(args.model, args.image)
        print(f"Prediction: {result['condition']} with {result['probabilities'][result['condition']]:.4f} confidence")
    
    print("Process completed successfully")


if __name__ == "__main__":
    main()
