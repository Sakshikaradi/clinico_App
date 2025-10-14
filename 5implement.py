import os
import urllib.request
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    # Only download if file doesn't exist
    if not os.path.exists(model_path):
        # Direct download link from Google Drive
        url = "https://drive.google.com/uc?export=download&id=import os
import urllib.request
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    # Only download if file doesn't exist
    if not os.path.exists(model_path):
        # Direct download link from Google Drive
        url = "https://drive.google.com/uc?export=download&id=1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0"
        st.info("Downloading model (~100 MB)... Please wait.")
        urllib.request.urlretrieve(url, model_path)
        st.success("Download completed!")

    # Verify file size (optional, prevent HTML downloads)
    if os.path.getsize(model_path) < 1000:  # less than 1 KB likely HTML page
        raise ValueError(
            "Downloaded file seems too small. Check the Google Drive link!"
        )

    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal / Pneumonia

    # Load weights safely
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        raise RuntimeError(
            f"Error loading the model. The .pth file may be corrupted: {e}"
        )

    model.eval()
    return model

# Load the model
model = load_model()

import urllib.request
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "model_finetuned.pth"

    # Only download if file doesn't exist
    if not os.path.exists(model_path):
        # Direct download link from Google Drive
        url = "https://drive.google.com/uc?export=download&id=1b2MVNoOAKrkV9wO4amuWPj4BTH3LvlY0"
        st.info("Downloading model (~100 MB)... Please wait.")
        urllib.request.urlretrieve(url, model_path)
        st.success("Download completed!")

    # Verify file size (optional, prevent HTML downloads)
    if os.path.getsize(model_path) < 1000:  # less than 1 KB likely HTML page
        raise ValueError(
            "Downloaded file seems too small. Check the Google Drive link!"
        )

    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal / Pneumonia

    # Load weights safely
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        raise RuntimeError(
            f"Error loading the model. The .pth file may be corrupted: {e}"
        )

    model.eval()
    return model

# Load the model
model = load_model()

        st.info("Downloading model (~100 MB)... Please wait.")
        urllib.request.urlretrieve(url, model_path)
        st.success("Download completed!")

    # Verify file size (optional, prevent HTML downloads)
    if os.path.getsize(model_path) < 1000:  # less than 1 KB likely HTML page
        raise ValueError(
            "Downloaded file seems too small. Check the Google Drive link!"
        )

    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal / Pneumonia

    # Load weights safely
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        raise RuntimeError(
            f"Error loading the model. The .pth file may be corrupted: {e}"
        )

    model.eval()
    return model

# Load the model
model = load_model()
