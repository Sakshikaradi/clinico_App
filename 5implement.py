import os
import urllib.request
import torch
from torchvision import models
import torch.nn as nn

def load_model():
    model_path = "model_finetuned.pth"
    url = "https://drive.google.com/file/d/1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0/view?usp=drive_link"
    # Download the model if not exists
    if not os.path.exists(model_path):
        print("Downloading model...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")

    # Initialize the model architecture
    model = models.resnet18(weights=None)  # No pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for 2 classes

    # ✅ Load model weights safely (PyTorch ≥2.6)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    except Exception as e:
        # If weights_only=False fails, fallback to older method
        print("Warning: weights_only=False failed. Attempting legacy load...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()  # Set to evaluation mode
    return model

# Usage in Streamlit
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









