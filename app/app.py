
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake & Cross-Age Face Verification",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# 2. PROJECT PATHS AND ARCFACE IMPORT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEEPFAKE_MODEL_PATH = (
    PROJECT_DIR / "models" / "efficientnet_b0_deepfake_best.pth"
)

# Allow Python to import arcface_verification.py from the
# project root, even though this app is inside the app folder.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arcface_verification import (
    verify_faces,
    SIMILARITY_THRESHOLD
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

CLASS_NAMES = ["Fake", "Real"]


# ============================================================
# 3. LOAD PYTORCH DEEPFAKE MODEL
# ============================================================

@st.cache_resource
def load_deepfake_model():
    """Load the trained EfficientNet-B0 deepfake checkpoint."""

    if not DEEPFAKE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Deepfake checkpoint not found: "
            f"{DEEPFAKE_MODEL_PATH}"
        )

    checkpoint = torch.load(
        DEEPFAKE_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint.get("architecture") != "efficientnet_b0":
        raise ValueError(
            "The checkpoint architecture is not EfficientNet-B0."
        )

    class_to_idx = checkpoint.get("class_to_idx")

    if class_to_idx != {"Fake": 0, "Real": 1}:
        raise ValueError(
            f"Unexpected class mapping in checkpoint: {class_to_idx}"
        )

    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 4. DEEPFAKE IMAGE PREPROCESSING
# ============================================================

deepfake_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def prepare_deepfake_image(uploaded_file):
    """
    Apply the same image transformation used by the
    existing EfficientNet-B0 inference pipeline.
    """

    image = Image.open(uploaded_file).convert("RGB")

    image_tensor = deepfake_transform(image)

    # Shape: [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


# ============================================================
# 5. DEEPFAKE PREDICTION
# ============================================================

def predict_deepfake(model, uploaded_file):
    """Return the predicted class and softmax class scores."""

    image_tensor = prepare_deepfake_image(uploaded_file)

    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    fake_score = float(probabilities[0].item())
    real_score = float(probabilities[1].item())

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    prediction = CLASS_NAMES[predicted_index]

    return prediction, fake_score, real_score


# ============================================================
# 6. MODEL STATUS
# ============================================================

deepfake_exists = DEEPFAKE_MODEL_PATH.exists()


# ============================================================
# 7. APPLICATION TITLE
# ============================================================

st.title(
    "🔍 Deepfake Detection & Cross-Age Face Verification"
)

st.write(
    "An academic project combining image-based deepfake "
    "classification with ArcFace-based cross-age face verification."
)

st.caption(
    f"Deepfake model device: {DEVICE}"
)

st.divider()


# ============================================================
# 8. MODEL STATUS DISPLAY
# ============================================================

col1, col2 = st.columns(2)

with col1:
    if deepfake_exists:
        st.success("✓ EfficientNet-B0 Deepfake Model Ready")
    else:
        st.error("✗ Deepfake checkpoint not found")

with col2:
    st.success("✓ ArcFace Verification Module Configured")

st.caption(
    "ArcFace is initialized when the verification module is imported. "
    "The first face comparison may take longer while models initialize."
)


# ============================================================
# 9. APPLICATION TABS
# ============================================================

tab1, tab2 = st.tabs([
    "🎭 Deepfake Detection",
    "👥 Cross-Age Verification"
])


# ============================================================
# 10. DEEPFAKE DETECTION TAB
# ============================================================

with tab1:

    st.header("Deepfake Face Detection")

    st.write(
        "Upload a face image to obtain the model's "
        "Fake or Real classification."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="deepfake_upload"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image.getvalue(),
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button(
            "Detect Deepfake",
            type="primary",
            key="detect_deepfake"
        ):

            if not deepfake_exists:
                st.error(
                    "Deepfake checkpoint is missing. "
                    "Check the models folder."
                )

            else:
                try:
                    with st.spinner("Analyzing image..."):

                        # Reset the file position before reading.
                        uploaded_image.seek(0)

                        model = load_deepfake_model()

                        prediction, fake_score, real_score = (
                            predict_deepfake(
                                model,
                                uploaded_image
                            )
                        )

                    st.subheader(
                        f"Prediction: {prediction}"
                    )

                    if prediction.upper() == "FAKE":
                        st.error(
                            "The model classified this image as FAKE."
                        )
                    else:
                        st.success(
                            "The model classified this image as REAL."
                        )

                    col_fake, col_real = st.columns(2)

                    with col_fake:
                        st.metric(
                            "Fake class score",
                            f"{fake_score * 100:.2f}%"
                        )

                    with col_real:
                        st.metric(
                            "Real class score",
                            f"{real_score * 100:.2f}%"
                        )

                    st.caption(
                        "These percentages are softmax class scores, "
                        "not calibrated probabilities or guarantees "
                        "that the image is genuine or manipulated."
                    )

                    st.progress(
                        max(0.0, min(1.0, fake_score)),
                        text="Fake class score"
                    )

                except Exception as error:
                    st.error("Could not run deepfake detection.")
                    st.exception(error)


# ============================================================
# 11. CROSS-AGE FACE VERIFICATION TAB — ARCFACE
# ============================================================

with tab2:

    st.header("Cross-Age Face Verification")

    st.write(
        "Upload two face images to estimate whether they "
        "may belong to the same person, even across different ages."
    )

    st.info(
        f"Cosine similarity threshold: {SIMILARITY_THRESHOLD:.6f}\n\n"
        "A similarity equal to or above this threshold is classified "
        "as Same person. A lower score is classified as Different people."
    )

    col1, col2 = st.columns(2)

    with col1:
        image1_file = st.file_uploader(
            "Upload first face image",
            type=["jpg", "jpeg", "png"],
            key="arcface_face1"
        )

    with col2:
        image2_file = st.file_uploader(
            "Upload second face image",
            type=["jpg", "jpeg", "png"],
            key="arcface_face2"
        )

    if image1_file is not None:
        st.image(
            image1_file.getvalue(),
            caption="First face image",
            use_container_width=True
        )

    if image2_file is not None:
        st.image(
            image2_file.getvalue(),
            caption="Second face image",
            use_container_width=True
        )

    if image1_file is not None and image2_file is not None:

        if st.button(
            "Verify Faces with ArcFace",
            type="primary",
            key="arcface_verify"
        ):

            try:
                with st.spinner(
                    "Detecting faces and comparing embeddings..."
                ):

                    # InsightFace's existing verification function
                    # accepts file paths, so save uploads temporarily.
                    with tempfile.TemporaryDirectory() as temp_dir:

                        image1_path = Path(temp_dir) / "face1.jpg"
                        image2_path = Path(temp_dir) / "face2.jpg"

                        image1_path.write_bytes(
                            image1_file.getvalue()
                        )

                        image2_path.write_bytes(
                            image2_file.getvalue()
                        )

                        result = verify_faces(
                            image1_path,
                            image2_path
                        )

                if not result["success"]:

                    st.warning(result["message"])

                    st.info(
                        "Try clearer images with a visible, "
                        "front-facing face."
                    )

                else:
                    similarity = result["similarity"]
                    prediction = result["prediction"]
                    threshold = result["threshold"]

                    st.subheader("Verification Result")

                    st.metric(
                        "Cosine Similarity",
                        f"{similarity:.4f}"
                    )

                    st.caption(
                        f"Decision threshold: {threshold:.6f}"
                    )

                    # Display the raw cosine similarity on a
                    # bounded scale; it is not a probability.
                    display_score = max(
                        0.0,
                        min(1.0, similarity)
                    )

                    st.progress(
                        display_score,
                        text="Cosine similarity (display scale)"
                    )

                    if prediction == "Same person":
                        st.success(
                            "✓ SAME PERSON — similarity meets "
                            "or exceeds the configured threshold."
                        )
                    else:
                        st.warning(
                            "DIFFERENT PEOPLE — similarity is below "
                            "the configured threshold."
                        )

                    st.caption(
                        "This result is a model estimate, not proof "
                        "of identity. ArcFace can make mistakes, "
                        "especially with large age gaps, poor image "
                        "quality, or multiple faces. The largest "
                        "detected face is selected automatically."
                    )

            except Exception as error:
                st.error("Could not run ArcFace verification.")
                st.exception(error)


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic Project — EfficientNet-B0 Deepfake Detection "
    "and ArcFace Cross-Age Face Verification"
)
