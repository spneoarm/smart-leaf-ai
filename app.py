import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


# Must match the exact training class order
CLASSES = [
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

MODEL_PATH = "models/leaf_model.pth"

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "leaf_model.pth"

st.set_page_config(page_title="Smart Leaf AI", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fff8 0%, #eef9f0 100%);
    }
    .hero-card {
        padding: 1.4rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.10);
        margin-bottom: 1rem;
    }
    .soft-card {
        padding: 1rem 1rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e6efe8;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
    }
    .result-card {
        padding: 1rem 1rem;
        border-radius: 16px;
        background: #f7fff7;
        border: 1px solid #dbeedd;
    }
    .small-muted {
        color: #5f6b61;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0;">Smart Leaf AI 🌿</h1>
        <p style="margin:0.5rem 0 0 0; font-size:1.05rem;">
            Upload a potato or tomato leaf image to predict whether it is healthy or diseased.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='small-muted'>Built with PyTorch + ResNet18 + Streamlit</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
# Cache the model so it only loads once.
def load_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

try:
    model = load_model()
    st.toast("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("Upload Leaf Image")
    st.write("Choose a clear image of a potato or tomato leaf.")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<p class='small-muted'>Tip: use a close-up image with good lighting and a visible leaf surface.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("Supported Classes")
    st.write("This demo currently predicts these categories:")
    for cls in CLASSES:
        pretty_cls = cls.replace("___", " - ").replace("_", " ")
        st.write(f"• {pretty_cls}")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a potato or tomato leaf image to see a prediction.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

        pretty_label = CLASSES[predicted_class].replace("___", " - ").replace("_", " ")

        preview_col, result_col = st.columns([1.05, 1], gap="large")

        with preview_col:
            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            st.subheader("Image Preview")
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            st.metric("Predicted Class", pretty_label)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(min(max(int(confidence), 0), 100))

            st.write("### Class Probabilities")
            for idx, prob in enumerate(probabilities[0]):
                label = CLASSES[idx].replace("___", " - ").replace("_", " ")
                pct = prob.item() * 100
                st.write(f"**{label}:** {pct:.2f}%")
                st.progress(min(max(int(pct), 0), 100))
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.caption("Demo project for plant disease classification using computer vision.")