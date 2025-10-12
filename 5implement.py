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
