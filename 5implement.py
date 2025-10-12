import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------
# 1️⃣ Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=False))
    model.eval()
    return model

# -------------------------------
# 2️⃣ Custom Grad-CAM
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

    # Gradients & activations
    gradients = model.layer4[1].conv2.weight.grad
    activations = model.layer4[1].conv2(img_tensor)

    # Weighted combination
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    gradcam = torch.sum(weights * activations, dim=1).squeeze().detach().cpu().numpy()

    # Normalize
    gradcam = np.maximum(gradcam, 0)
    gradcam /= gradcam.max() + 1e-8
    return gradcam

# -------------------------------
# 3️⃣ Overlay Heatmap on Image
# -------------------------------
def overlay_heatmap_on_image(image, heatmap):
    """
    image: PIL.Image
    heatmap: 2D numpy array normalized 0-1
    """
    # Resize heatmap to match image
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
    heatmap_rgb = np.array(heatmap_resized.convert("RGB"))

    # Convert original image to NumPy
    image_np = np.array(image)

    # Superimpose
    superimposed = (0.6 * image_np + 0.4 * heatmap_rgb).astype(np.uint8)
    return Image.fromarray(superimposed)

# -------------------------------
# 4️⃣ Streamlit App Layout
# -------------------------------
st.title("🩻 Chest X-Ray Classification with Grad-CAM (Cloud-Compatible)")
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
    gradcam_image = overlay_heatmap_on_image(image, gradcam)
    st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
