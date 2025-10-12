import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# ✅ 1. Model Loading
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=False))
    model.eval()
    return model

# -------------------------------
# ✅ 2. Grad-CAM (Custom Implementation)
# -------------------------------
def generate_gradcam(model, img_tensor, target_class=None):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad_()

    # Forward pass
    outputs = model(img_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    score = outputs[0, target_class]
    score.backward()

    # Extract gradients and activations from last conv layer
    gradients = model.layer4[1].conv2.weight.grad
    activations = model.layer4[1].conv2(img_tensor)

    # Weighted combination
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    gradcam = torch.sum(weights * activations, dim=1).squeeze().detach().cpu().numpy()

    # Normalize Grad-CAM
    gradcam = np.maximum(gradcam, 0)
    gradcam /= gradcam.max() + 1e-8
    return gradcam

# -------------------------------
# ✅ 3. Streamlit App Layout
# -------------------------------
st.title("🩻 Chest X-Ray Classification with Grad-CAM (Cloud Compatible)")
st.write("Upload a Chest X-ray image to classify as **Normal** or **Pneumonia** and visualize model focus areas.")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)

    # Load model
    model = load_model()

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probs).item()
        classes = ["Normal", "Pneumonia"]
        st.subheader(f"✅ Prediction: {classes[predicted_class]}")
        st.write(f"**Confidence:** {probs[predicted_class]*100:.2f}%")

    # Grad-CAM Visualization
    st.subheader("🧠 Model Focus (Grad-CAM)")
    gradcam = generate_gradcam(model, img_tensor, predicted_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    image_resized = np.array(image.resize((224, 224)))
    superimposed = cv2.addWeighted(cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    st.image(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB),
             caption="Grad-CAM Heatmap",
             use_container_width=True)
