import json
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.combined_model.Combined_Model import ModelAggregator
from torch import autocast


class OCTInference:
    """Class for OCT image inference with the trained model."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the inference class.
        Args:
            model_path: Optional path to model weights
        """
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.model = ModelAggregator()
        # Move model to inference device
        self.model.to(self.device)
        # Load weights if provided
        if model_path is not None:
            self._load_model(model_path)
        # Set model to evaluation mode
        self.model.eval()

        # Update class labels to match training data
        self.condition_names = {
            0: "Normal",
            1: "CNV (Choroidal Neovascularization)",
            2: "DME (Diabetic Macular Edema)",
            3: "Drusen",
        }

        # Fluid class colors and labels
        self.fluid_colors = {
            0: (0, 0, 0),  # Background (black)
            1: (255, 0, 0),  # Red fluid
            2: (0, 255, 0),  # Green fluid
            3: (0, 0, 255),  # Blue fluid
            4: (255, 192, 203),  # Pink fluid
            5: (255, 255, 0),  # Yellow fluid
        }

        # Fluid type legend
        self.fluid_legend = {
            "Background": "Background",
            "Red": "Sub-Retinal Fluid",
            "Green": "Intra-Retinal Fluid",
            "Blue": "Pigment Epithelial Detachment",
            "Pink": "Integrity Of Inner/Outer Segment",
            "Yellow": "Subretinal hyper-reflective material",
        }

        # Mapping from color index to fluid type
        self.fluid_index_to_name = {
            0: "Background",
            1: "Sub-Retinal Fluid",
            2: "Intra-Retinal Fluid",
            3: "Pigment Epithelial Detachment",
            4: "Integrity Of Inner/Outer Segment",
            5: "Subretinal hyper-reflective material",
        }

    def _load_model(self, model_path: Union[str, Path]) -> None:
        """Load model weights from file.

        Args:
            model_path: Path to model weights
        """
        try:
            if isinstance(model_path, str):
                model_path = Path(model_path)

            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def remove_oct_artifacts(self, image):
        """Remove common OCT artifacts like scan lines and shadows."""
        import cv2
        import numpy as np

        # Copy the image
        cleaned_image = image.copy()

        # 1. Remove horizontal scan lines using morphological operations
        # Create a horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

        # Detect horizontal lines
        detected_lines = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )

        # Remove these lines from the original image
        cleaned_image = cv2.subtract(cleaned_image, detected_lines)

        # 2. Handle shadow artifacts (vertical dark regions)
        # Calculate column-wise mean to detect shadows
        col_means = np.mean(cleaned_image, axis=0)

        # Smooth the means to detect significant drops
        smoothed_means = np.convolve(col_means, np.ones(10) / 10, mode="same")

        # Find columns with significantly lower intensity (potential shadows)
        global_mean = np.mean(smoothed_means)
        shadow_cols = np.where(smoothed_means < global_mean * 0.7)[0]

        # Correct shadow columns by interpolating from neighboring columns
        for col in shadow_cols:
            left_bound = max(0, col - 5)
            right_bound = min(cleaned_image.shape[1] - 1, col + 5)

            # Skip if we're at the image edge
            if left_bound == 0 or right_bound == cleaned_image.shape[1] - 1:
                continue

            # Find non-shadow columns for interpolation
            valid_cols = [
                i for i in range(left_bound, right_bound + 1) if i not in shadow_cols
            ]

            if len(valid_cols) > 1:
                # Interpolate values from valid columns
                for row in range(cleaned_image.shape[0]):
                    valid_values = [cleaned_image[row, i] for i in valid_cols]
                    cleaned_image[row, col] = np.mean(valid_values)

        return cleaned_image

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a retinal OCT scan for model input.

        Args:
            image: Input OCT image (H, W, 3) in RGB format or (H, W) in grayscale

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        import cv2

        # Make a copy to avoid modifying the original
        processed_image = image.copy()

        # Convert to grayscale if it's RGB (OCT scans are often more informative in grayscale)
        if len(processed_image.shape) == 3:
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = processed_image.copy()
        cleaned_image = self.remove_oct_artifacts(gray_image)
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(cleaned_image)

        # 2. Bilateral filtering - preserves edges while reducing noise
        bilateral_filtered = cv2.bilateralFilter(
            enhanced_image, d=9, sigmaColor=75, sigmaSpace=75
        )

        # 3. Wavelet denoising (PyWavelets library)
        try:
            import numpy as np
            import pywt

            # Wavelet transform
            coeffs = pywt.wavedec2(bilateral_filtered, "bior4.4", level=2)
            coeffs_array, coeff_slices = pywt.coeffs_to_array(coeffs)
            # Threshold coefficients to remove noise
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(bilateral_filtered.size))

            # Apply threshold
            new_coeffs = pywt.array_to_coeffs(
                coeffs_array, coeff_slices, output_format="wavedec2"
            )
            # Reconstruct image
            denoised_image = pywt.waverec2(new_coeffs, "bior4.4")

            # Ensure the reconstructed image has the right dimensions
            denoised_image = denoised_image[
                : bilateral_filtered.shape[0], : bilateral_filtered.shape[1]
            ]
            denoised_image = np.uint8(denoised_image)
        except ImportError:
            print("PyWavelets not available. Skipping wavelet denoising.")
            denoised_image = bilateral_filtered

        # 4. Speckle noise reduction (specific to OCT images)
        denoised_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)

        # 5. Standardize image size (OCT scans often need consistent dimensions)
        standard_size = (512, 512)  # Common size for OCT analysis
        resized_image = cv2.resize(
            denoised_image, standard_size, interpolation=cv2.INTER_CUBIC
        )

        # 6. Normalize intensity values
        normalized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)

        # 7. Convert back to RGB for model input (if needed)
        rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        # Convert to PyTorch tensor and add batch dimension
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Normalize to [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        return tensor

    def predict(
        self, image: Union[np.ndarray, torch.Tensor], return_segmentation: bool = False
    ) -> Dict:
        """Run inference on a single image.
        Args:
            image: Input image as numpy array or tensor
            return_segmentation: Whether to return segmentation masks
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image if needed
        if isinstance(image, np.ndarray):
            tensor = self.preprocess_image(image)
        else:
            tensor = image

        # Move to device
        tensor = tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            # Apply transformations for segmentation model
            seg_input = self.model.seg_transforms(tensor)

            # Get segmentation masks
            seg_output = self.model.segmentor(seg_input)["out"]

            # Apply transformations for classification model
            cls_input = self.model.transforms(tensor)

            # Resize masks to match classification input dimensions
            masks = F.interpolate(
                seg_output,
                size=cls_input.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            masks = F.softmax(masks, dim=1)

            # Combine for classification
            combined = torch.cat([cls_input, masks], dim=1)

            # Get classification output
            logits = self.model.classifier(combined)
            probs = F.softmax(logits, dim=1)

        # Get predicted class
        probs_np = probs.cpu().numpy()[0]
        predicted_class = int(probs_np.argmax())

        # Create result dictionary
        result = {
            "condition": self.condition_names[predicted_class],
            "condition_id": predicted_class,
            "probabilities": {
                self.condition_names[i]: float(probs_np[i])
                for i in range(len(probs_np))
            },
        }

        # Add segmentation if requested
        if return_segmentation:
            # Convert masks to class indices
            seg_masks = masks.cpu().numpy()[0]  # (C, H, W)
            seg_classes = np.argmax(seg_masks, axis=0)  # (H, W)

            # Calculate fluid percentages
            total_pixels = seg_classes.size
            fluid_percentages = {
                self.fluid_index_to_name[i]: float(
                    np.sum(seg_classes == i) / total_pixels * 100
                )
                for i in range(min(len(self.fluid_index_to_name), seg_masks.shape[0]))
            }

            result["segmentation"] = {
                "masks": seg_classes.tolist(),
                "fluid_percentages": fluid_percentages,
            }

        return result

    def visualize_prediction(
        self,
        image: np.ndarray,
        prediction: Dict,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Visualize prediction results on the image.
        Args:
            image: Input image
            prediction: Prediction dictionary from predict()
            save_path: Optional path to save visualization
        Returns:
            Visualization image
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Display original image
            ax1.imshow(image)
            ax1.set_title("Original OCT Image")
            ax1.axis("off")

            # Create condition probability bar chart
            conditions = list(prediction["probabilities"].keys())
            probabilities = list(prediction["probabilities"].values())
            colors = ["green", "orange", "red", "purple"]
            bars = ax2.barh(conditions, probabilities, color=colors)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Probability")
            ax2.set_title(f'Predicted: {prediction["condition"]}')

            # Add probability values
            for bar, prob in zip(bars, probabilities):
                ax2.text(
                    min(prob + 0.05, 0.95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob:.2f}",
                    va="center",
                )

            # Add segmentation visualization if available
            if "segmentation" in prediction:
                fig.set_figheight(9)
                # Create third subplot for segmentation
                ax3 = fig.add_subplot(2, 1, 2)
                # Get segmentation mask
                seg_mask = np.array(prediction["segmentation"]["masks"])

                # Create colormap for segmentation - using the actual colors from fluid_colors
                colors_list = []
                for i in range(6):  # Assuming 6 classes (background + 5 fluid types)
                    rgb_color = self.fluid_colors[i]
                    # Convert RGB (0-255) to normalized RGB (0-1)
                    norm_color = [c / 255.0 for c in rgb_color]
                    colors_list.append(norm_color)

                cmap = LinearSegmentedColormap.from_list("fluid_cmap", colors_list, N=6)

                # Plot segmentation mask
                im = ax3.imshow(seg_mask, cmap=cmap, vmin=0, vmax=5)
                ax3.set_title("Fluid Segmentation")
                ax3.axis("off")

                # Add colorbar
                cbar = plt.colorbar(
                    im,
                    ax=ax3,
                    orientation="horizontal",
                    ticks=[0.4, 1.2, 2.0, 2.8, 3.6, 4.4],
                )

                # Use the correct fluid type labels for the colorbar
                cbar.set_ticklabels(
                    [
                        "Background",
                        "Sub-Retinal Fluid",
                        "Intra-Retinal Fluid",
                        "Pigment Epithelial Detachment",
                        "Integrity Of Inner/Outer Segment",
                        "Subretinal hyper-reflective material",
                    ]
                )

                # Add fluid percentages
                text_str = "Fluid Percentages:\n"
                for fluid, percentage in prediction["segmentation"][
                    "fluid_percentages"
                ].items():
                    if (
                        fluid != "Background" and percentage > 0.01
                    ):  # Skip background and very small percentages
                        text_str += f"{fluid}: {percentage:.2f}%\n"

                ax3.text(
                    1.05,
                    0.5,
                    text_str,
                    transform=ax3.transAxes,
                    fontsize=9,
                    verticalalignment="center",
                )

            plt.tight_layout()

            # Save if path provided
            if save_path:
                if isinstance(save_path, str):
                    save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Visualization saved to {save_path}")

            # Return the figure to allow further customization
            return fig

        except ImportError:
            print("Visualization requires matplotlib")
            return np.array(None)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None


def load_and_run_inference(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = "inference_results",
) -> Dict:
    """Load model and run inference on a single image.

    Args:
        model_path: Path to model weights
        image_path: Path to input image
        output_dir: Directory to save results

    Returns:
        Prediction results
    """
    try:
        import cv2

        # Create inference object
        inference = OCTInference(model_path)

        # Load image
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run prediction
        prediction = inference.predict(image, return_segmentation=True)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction as JSON
        results_path = output_dir / f"{image_path.stem}_results.json"
        with open(results_path, "w") as f:
            json.dump(prediction, f, indent=4)

        # Create and save visualization
        vis_path = output_dir / f"{image_path.stem}_visualization.png"
        inference.visualize_prediction(image, prediction, vis_path)

        return prediction

    except ImportError:
        print("OpenCV is required for image loading")
        return {}
    except Exception as e:
        print(f"Error during inference: {e}")
        return {}


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import sys
    from pathlib import Path

    import cv2
    import matplotlib
    import numpy as np

    # Use non-interactive backend to avoid XCB issues
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="OCT Image Inference Tool")
    parser.add_argument(
        "--model",
        type=str,
        default="models/combined_model/weights/best_model_epoch_3.pth",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--image_dir", type=str, help="Directory containing OCT images to analyze"
    )
    parser.add_argument(
        "--image", type=str, help="Path to a single OCT image to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save inference results",
    )
    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    # Initialize model
    try:
        print(f"Loading model from {args.model}...")
        model = OCTInference(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # Process images
    image_paths = []

    # Handle single image or directory of images
    if args.image:
        if os.path.exists(args.image):
            image_paths = [args.image]
        else:
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
    elif args.image_dir:
        if os.path.isdir(args.image_dir):
            image_paths = []
            for ext in ["jpg", "jpeg", "png", "tif", "tiff"]:
                image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
                image_paths.extend(
                    glob.glob(os.path.join(args.image_dir, f"*.{ext.upper()}"))
                )
            if not image_paths:
                print(f"No images found in {args.image_dir}")
                sys.exit(1)
            print(f"Found {len(image_paths)} images in directory")
        else:
            print(f"Error: Directory not found at {args.image_dir}")
            sys.exit(1)
    else:
        # Interactive selection if no image or directory provided
        print("\nNo image specified. Let's browse for images:")
        print("Enter the path to an OCT image or directory with OCT images:")
        path = input("> ").strip()

        if os.path.isdir(path):
            # It's a directory, list all images
            image_paths = []
            for ext in ["jpg", "jpeg", "png", "tif", "tiff"]:
                image_paths.extend(glob.glob(os.path.join(path, f"*.{ext}")))
                image_paths.extend(glob.glob(os.path.join(path, f"*.{ext.upper()}")))

            if not image_paths:
                print(f"No images found in {path}")
                sys.exit(1)

            # List found images for selection
            print(f"\nFound {len(image_paths)} images. Select an image by number:")
            for i, img_path in enumerate(image_paths):
                print(f"[{i+1}] {os.path.basename(img_path)}")

            try:
                selection = int(input("\nEnter image number (or 0 to process all): "))
                if selection == 0:
                    # Process all images
                    print(f"Processing all {len(image_paths)} images...")
                else:
                    # Process single selected image
                    image_paths = [image_paths[selection - 1]]
                    print(f"Selected: {os.path.basename(image_paths[0])}")
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                sys.exit(1)
        elif os.path.isfile(path):
            # It's a single file
            image_paths = [path]
            print(f"Selected: {os.path.basename(path)}")
        else:
            print(f"Error: Path not found: {path}")
            sys.exit(1)

    # Process each image
    for img_path in image_paths:
        try:
            print(f"\nProcessing: {os.path.basename(img_path)}")

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run prediction
            print("Running inference...")
            prediction = model.predict(image, return_segmentation=True)

            # Save results
            filename = os.path.basename(img_path)
            filename_stem = os.path.splitext(filename)[0]
            results_path = os.path.join(
                args.output_dir, f"{filename_stem}_results.json"
            )
            vis_path = os.path.join(
                args.output_dir, f"{filename_stem}_visualization.png"
            )

            with open(results_path, "w") as f:
                json.dump(prediction, f, indent=4)

            # Print results
            print(f"\nResults for {filename}:")
            print(f"Predicted condition: {prediction['condition']}")
            print("\nProbabilities:")
            for condition, prob in prediction["probabilities"].items():
                print(f"  {condition}: {prob:.4f}")

            if "segmentation" in prediction:
                print("\nFluid Percentages:")
                for fluid, percentage in prediction["segmentation"][
                    "fluid_percentages"
                ].items():
                    print(f"  {fluid}: {percentage:.2f}%")

            # Generate and save visualization
            print(f"\nGenerating visualization...")
            model.visualize_prediction(image, prediction, vis_path)
            print(f"Visualization saved to {vis_path}")
            print(f"JSON results saved to {results_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\nProcessing complete!")
