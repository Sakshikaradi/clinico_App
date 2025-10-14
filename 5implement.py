import os
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------
# 1. Load fine-tuned model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    # Download the model if it does not exist
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0"
        urllib.request.urlretrieve(url, model_path)

    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal / Pneumonia

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Load the model
model = load_model()

# -------------------------------
# 2. Image transforms
# -------------------------------
def transform_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# -------------------------------
# 3. Streamlit interface
# -------------------------------
st.title("CliniScan: Lung-Abnormality Detection")
st.write("Upload a chest X-ray image to detect Normal vs Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform_image(image)

    # -------------------------------
    # 4. Prediction
    # -------------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        classes = ["Normal", "Pneumonia"]
        predicted_class = classes[torch.argmax(probs)]
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Class Probabilities: Normal={probs[0,0]:.3f}, Pneumonia={probs[0,1]:.3f}")

    # -------------------------------
    # 5. Grad-CAM visualization
    # -------------------------------
    target_layers = [model.layer4[-1]]  # Last conv layer of ResNet18
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    rgb_image = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
