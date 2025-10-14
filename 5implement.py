import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------
# 1. Load fine-tuned model
# -------------------------------

import urllib.request

@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    if not os.path.exists(model_path):
        url =https://drive.google.com/file/d/1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0/view?usp=sharing  # e.g. a public Google Drive or Dropbox link
        urllib.request.urlretrieve(url, model_path)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# -------------------------------
# 2. Image transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🩻 Chest X-Ray Classification with Grad-CAM")
st.write("Upload a Chest X-ray image to classify as Normal or Pneumonia and visualize model attention.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    classes = ["Normal", "Pneumonia"]
    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx] * 100

    st.warning(f"⚠️ Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # -------------------------------
    # Class Probabilities (move inside block)
    # -------------------------------
    st.subheader("Class Probabilities")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[i]:.4f}")
        st.progress(int(probs[i] * 100))  # convert to integer percentage

    # -------------------------------
    # Grad-CAM Visualization
    # -------------------------------
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    rgb_img = np.array(img.resize((224, 224))) / 255.0
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.subheader("Grad-CAM Visualization")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.imshow(visualization)
    plt.axis("off")
    st.pyplot(plt)

