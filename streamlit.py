import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

# Existing OCTInference class (assuming in imports)
from models.combined_model.inference import OCTInference
from PIL import Image
from torch import nn
from torchvision.models import VGG11_BN_Weights, vgg11_bn


def _init_model():
    """Initialize VGG11 model with proper transforms"""
    from torchvision.models import VGG11_BN_Weights, vgg11_bn

    # Load pretrained weights and transforms
    weights = VGG11_BN_Weights.DEFAULT
    transforms = weights.transforms()

    # Create model with modified classifier
    model = vgg11_bn(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify classifier
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, 4)

    return transforms, model


class VGG11Classifier:
    def __init__(self, model_path):
        self.trans, self.model = _init_model()
        self.model.load_state_dict(
            torch.load(
                "./models/classification_model/saved_models/classification_model_5.pth"
            )
        )
        self.model.eval()
        self.class_names = ["CNV", "DME", "Drusen", "Normal"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, image):
        """Make prediction on an input image (numpy array in RGB format)"""
        with torch.no_grad():
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)

            # Apply transformations and add batch dimension
            inputs = self.trans(pil_image).unsqueeze(0).to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

            return {
                "condition": self.class_names[torch.argmax(probs)],
                "probabilities": {
                    self.class_names[i]: probs[i].item() for i in range(4)
                },
            }


class CombinedModel:
    def __init__(self, seg_model_path, cls_model_path):
        self.seg_model = OCTInference(seg_model_path)
        self.cls_model = vgg11(pretrained=False)
        self.cls_model.classifier[6] = torch.nn.Linear(4096, 4)
        self.cls_model.load_state_dict(torch.load(cls_model_path))
        self.cls_model.eval()
        self.class_names = ["Normal", "CNV", "DME", "Drusen"]

    def preprocess_classifier(self, image):
        img = Image.fromarray(image).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

    def predict(self, image):
        # Get segmentation results
        seg_results = self.seg_model.predict(image, return_segmentation=True)

        # Get classifier results
        with torch.no_grad():
            inputs = self.preprocess_classifier(image)
            outputs = self.cls_model(inputs)
            cls_probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Combine probabilities using weighted average
        combined_probs = {
            cls: (
                seg_results["probabilities"].get(cls, 0) * 0.7
                + cls_probs[i].item() * 0.3
            )
            for i, cls in enumerate(self.class_names)
        }

        return {
            "segmentation": seg_results,
            "classification": {
                "condition": self.class_names[torch.argmax(cls_probs)],
                "probabilities": {
                    self.class_names[i]: cls_probs[i].item() for i in range(4)
                },
            },
            "combined": {
                "condition": max(combined_probs, key=combined_probs.get),
                "probabilities": combined_probs,
            },
        }


# Configure page settings
st.set_page_config(page_title="OCT Analysis Dashboard", page_icon="👁️", layout="wide")

# Custom CSS for styling
st.markdown(
    """
<style>
    .stProgress > div > div > div > div {
        background-color: #1E90FF;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0E1117;
    }
    .st-cj {
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_type = None

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Model Configuration")

    # Model selection
    model_type = st.radio(
        "Select Model Type",
        [
            "Combined Model (Segmentation+Classification)",
            "Baseline Classifier (VGG11)",
            "Ensemble Model (Combined + Classifier)",
        ],
    )

    model_paths = {
        "Combined Model (Segmentation+Classification)": "models/combined_model/weights/best_model_epoch_3.pth",
        "Baseline Classifier (VGG11)": "./models/combined_model/weights/classification_model_best.pth",
        "Ensemble Model (Combined + Classifier)": (
            "models/combined_model/weights/best_model_epoch_3.pth",
            "models/classification_model_best.pth",
        ),
    }

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                if model_type == "Baseline Classifier (VGG11)":
                    st.session_state.model = VGG11Classifier(model_paths[model_type])
                elif model_type == "Ensemble Model (Combined + Classifier)":
                    st.session_state.model = CombinedModel(*model_paths[model_type])
                else:
                    st.session_state.model = OCTInference(model_paths[model_type])

                st.session_state.model_type = model_type
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

    st.markdown("---")
    st.subheader("🔍 Fluid Type Legend")
    fluid_legend = {
        "🔴 Red": "Sub-Retinal Fluid",
        "🟢 Green": "Intra-Retinal Fluid",
        "🔵 Blue": "Pigment Epithelial Detachment",
        "🌸 Pink": "Integrity Of Inner/Outer Segment",
        "🟡 Yellow": "Subretinal hyper-reflective material",
    }
    for color, desc in fluid_legend.items():
        st.markdown(f"{color}: {desc}")

# Main content area
st.title("👁️ OCT Image Analysis Dashboard")
st.markdown("Drag and drop an OCT image below to analyze")

uploaded_file = st.file_uploader(
    "Upload OCT Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    label_visibility="hidden",
)


def visualize_segmentation(seg_mask):
    """Convert segmentation mask to RGB image for visualization"""
    seg_rgb = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    color_map = {
        0: [0, 0, 0],  # Background
        1: [255, 0, 0],  # Red
        2: [0, 255, 0],  # Green
        3: [0, 0, 255],  # Blue
        4: [255, 192, 203],  # Pink
        5: [255, 255, 0],  # Yellow
    }
    for class_id, color in color_map.items():
        seg_rgb[seg_mask == class_id] = color
    return seg_rgb


if uploaded_file is not None and st.session_state.model:
    # Read and process image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_array, use_column_width=True)

    # Run prediction
    with st.spinner("Analyzing image..."):
        if st.session_state.model_type == "Ensemble Model (Combined + Classifier)":
            prediction = st.session_state.model.predict(img_array)
        elif st.session_state.model_type == "Baseline Classifier (VGG11)":
            prediction = st.session_state.model.predict(img_array)
        else:
            prediction = st.session_state.model.predict(
                img_array, return_segmentation=True
            )
    # Display results based on model type
    if st.session_state.model_type == "Ensemble Model (Combined + Classifier)":
        # Ensemble model display
        with col2:
            st.subheader("Combined Diagnosis")
            comb_pred = prediction["combined"]
            st.markdown(
                f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background: #1E1E1E;
                margin-bottom: 20px;
            ">
                <h3 style="margin:0;color:#1E90FF">{comb_pred['condition']}</h3>
                <p style="margin:0;color:#888">Confidence: {max(comb_pred['probabilities'].values()):.2%}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            fig = px.bar(
                pd.DataFrame(
                    {
                        "Condition": comb_pred["probabilities"].keys(),
                        "Confidence": comb_pred["probabilities"].values(),
                    }
                ),
                x="Confidence",
                y="Condition",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segmentation visualization
        st.subheader("Fluid Analysis")
        col3, col4 = st.columns(2)

        with col3:
            seg_mask = np.array(prediction["segmentation"]["masks"])
            seg_rgb = visualize_segmentation(seg_mask)
            st.image(seg_rgb, caption="Segmentation Mask", use_column_width=True)

        with col4:
            fluid_data = prediction["segmentation"]["fluid_percentages"]
            fluid_df = pd.DataFrame(
                {"Fluid Type": fluid_data.keys(), "Percentage": fluid_data.values()}
            ).query('`Fluid Type` != "Background" and Percentage > 0.1')

            if not fluid_df.empty:
                fig = px.pie(
                    fluid_df,
                    names="Fluid Type",
                    values="Percentage",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant fluid detected")

        # Classifier results
        st.subheader("Component Model Results")
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("**Segmentation-Based Diagnosis**")
            seg_pred = prediction["segmentation"]
            fig = px.bar(
                pd.DataFrame(
                    {
                        "Condition": seg_pred["probabilities"].keys(),
                        "Confidence": seg_pred["probabilities"].values(),
                    }
                ),
                x="Confidence",
                y="Condition",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.markdown("**VGG11 Classifier Diagnosis**")
            cls_pred = prediction["classification"]
            fig = px.bar(
                pd.DataFrame(
                    {
                        "Condition": cls_pred["probabilities"].keys(),
                        "Confidence": cls_pred["probabilities"].values(),
                    }
                ),
                x="Confidence",
                y="Condition",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.model_type == "Baseline Classifier (VGG11)":
        with col2:
            st.subheader("Diagnosis Results")
            condition = prediction["condition"]
            confidence = max(prediction["probabilities"].values())
            st.markdown(
                f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background: #1E1E1E;
                margin-bottom: 20px;
            ">
                <h3 style="margin:0;color:#1E90FF">{condition}</h3>
                <p style="margin:0;color:#888">Confidence: {confidence:.2%}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            fig = px.bar(
                pd.DataFrame(
                    {
                        "Condition": prediction["probabilities"].keys(),
                        "Confidence": prediction["probabilities"].values(),
                    }
                ),
                x="Confidence",
                y="Condition",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Original Combined Model
        with col2:
            st.subheader("Diagnosis Results")
            condition = prediction["condition"]
            confidence = max(prediction["probabilities"].values())
            st.markdown(
                f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background: #1E1E1E;
                margin-bottom: 20px;
            ">
                <h3 style="margin:0;color:#1E90FF">{condition}</h3>
                <p style="margin:0;color:#888">Confidence: {confidence:.2%}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            fig = px.bar(
                pd.DataFrame(
                    {
                        "Condition": prediction["probabilities"].keys(),
                        "Confidence": prediction["probabilities"].values(),
                    }
                ),
                x="Confidence",
                y="Condition",
                orientation="h",
                color="Confidence",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segmentation visualization
        st.subheader("Fluid Analysis")
        col3, col4 = st.columns(2)

        with col3:
            seg_mask = np.array(prediction["segmentation"]["masks"])
            seg_rgb = visualize_segmentation(seg_mask)
            st.image(seg_rgb, caption="Segmentation Mask", use_column_width=True)

        with col4:
            fluid_data = prediction["segmentation"]["fluid_percentages"]
            fluid_df = pd.DataFrame(
                {"Fluid Type": fluid_data.keys(), "Percentage": fluid_data.values()}
            ).query('`Fluid Type` != "Background" and Percentage > 0.1')

            if not fluid_df.empty:
                fig = px.pie(
                    fluid_df,
                    names="Fluid Type",
                    values="Percentage",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant fluid detected")

    # Show raw data
    with st.expander("View Raw Prediction Data"):
        st.json(prediction)

elif uploaded_file and not st.session_state.model:
    st.warning("⚠️ Please load the model first in the sidebar")
