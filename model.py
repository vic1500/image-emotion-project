import cv2
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import torch
import streamlit as st

@st.cache_resource
def load_model(model_path):
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    return processor, model


def predict_emotion(img, model, processor):
    if isinstance(img, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # If img is a file path (string)
    elif isinstance(img, str):
        image = Image.open(img)
    # If img is a file-like object (e.g., UploadedFile)
    else:
        image = Image.open(img)

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotions[predicted_class]
    confidence = predictions[0][predicted_class].item()

    return predicted_emotion, confidence
