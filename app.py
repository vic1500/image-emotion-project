import streamlit as st
import sqlite3
from PIL import Image
from model import predict_emotion, load_model

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

st.title("😊 Image Emotion Detector")
st.write("Upload a picture and I'll tell you what emotion it shows!")

MODEL_PATH = "abhilash88/face-emotion-detection"
DB_PATH = "image_emotion.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

processor, model = load_model(MODEL_PATH)

def create_table():
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS image_emotion(
            id INTEGER PRIMARY KEY,
            image_name TEXT,
            detected_emotion TEXT,
            confidence REAL
        )"""
    )

def save_values_to_table(name, emotion, confidence):
    cursor.execute("INSERT INTO image_emotion(image_name, detected_emotion, confidence) VALUES (?, ?, ?)", (name, emotion, confidence))
    conn.commit()

create_table()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

emotions_emoji = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

if uploaded_file is not None:

    img_col, result_col = st.columns(2)

    with img_col:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    predict_btn = st.button("Predict Emotion", type="primary")

    if predict_btn:
        with result_col:

            results = predict_emotion(uploaded_file, model, processor)

            if results:
                st.subheader("Detection Results")
                emotion, score = results
                st.success(f"**Detected Emotion:** {emotion.capitalize()} {emotions_emoji[emotion.lower()]}")
                st.write(f"Confidence: {(score * 100):.2f}%")
                
                save_values_to_table(uploaded_file.name, emotion, score)

            else:
                st.warning("No face detected. Try another image.")
else:
    st.info("Please upload an image to begin.")

