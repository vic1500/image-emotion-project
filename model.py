from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_model(model_path):
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    return processor, model


def predict_emotion(img, model, processor):
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
