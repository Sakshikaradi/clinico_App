import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st
from PIL import Image
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------
# 1. Load fine-tuned model
# -------------------------------

@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    # Download if missing
    if not os.path.exists(model_path):
        import gdown
        url = "https://drive.google.com/uc?id=1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0"
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

    # Load architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model



# Load the model
model = load_model()

# -------------------------------
# 2. Define image transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

st.title("🩻 Chest X-Ray Classification (Normal vs Pneumonia)")

# -------------------------------
# 3. Upload and predict
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        classes = ["Normal", "Pneumonia"]
        st.subheader(f"🩺 Prediction: **{classes[predicted.item()]}**")



