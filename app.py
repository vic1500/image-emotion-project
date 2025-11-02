import io
from datetime import datetime
import sqlite3
import time
import uuid
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from model import predict_emotion, load_model

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

st.title("😊 Image Emotion Detector")
st.write("Upload a picture and I'll tell you what emotion it shows!")

image_upload_tab, live_caption_tab, history_tab = st.tabs(["Image Upload", "Live Caption", "History"])

MODEL_PATH = "abhilash88/face-emotion-detection"
DB_PATH = "image_emotion.db"

emotions_emoji = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

processor, model = load_model(MODEL_PATH)

def create_table():
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS image_emotion(
            id INTEGER PRIMARY KEY UNIQUE,
            image_name TEXT,
            detected_emotion TEXT,
            confidence REAL,
            data BLOB,
            timestamp TEXT
        )"""
    )

def save_values_to_table(name, emotion, confidence, data, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
    cursor.execute("INSERT INTO image_emotion(image_name, detected_emotion, confidence, data, timestamp) VALUES (?, ?, ?, ?, ?)", (name, emotion, confidence, data, timestamp))
    conn.commit()

def get_connection():
    return sqlite3.connect("image_emotion.db", check_same_thread=False)

def delete_values_from_table(id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_emotion WHERE id=?", (id,))
        conn.commit()

create_table()

with image_upload_tab:

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])


    if uploaded_file is not None:

        blob = uploaded_file.getvalue()

        img_col, result_col = st.columns(2)

        with img_col:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')

        predict_btn = st.button("Predict Emotion", type="primary")

        if predict_btn:
            with result_col:

                results = predict_emotion(uploaded_file, model, processor)

                if results:
                    st.subheader("Detection Results")
                    emotion, score = results
                    st.success(f"**Detected Emotion:** {emotion.capitalize()} {emotions_emoji[emotion.lower()]}")
                    st.write(f"Confidence: {(score * 100):.2f}%")
                    
                    save_values_to_table(uploaded_file.name, emotion, score, blob)

                else:
                    st.warning("No face detected. Try another image.")
    else:
        st.info("Please upload an image to begin.")

with live_caption_tab:
    if "camera" not in st.session_state:
        st.session_state.camera = None
    if "last_frame" not in st.session_state:
        st.session_state.last_frame = None
    if "analyze_triggered" not in st.session_state:
        st.session_state.analyze_triggered = False

    # UI controls
    run = st.checkbox("Start Camera", value=False)
    start_camera = st.info("Start Camera to capture live image")
    stframe = st.empty()
    analyze_button_placeholder = st.empty()
    result_placeholder = st.empty()

    if run:
        start_camera.empty()
        # Start camera
        if st.session_state.camera is None:
            st.session_state.camera = cv2.VideoCapture(0)

        camera = st.session_state.camera

        # Display analyze button
        analyze = analyze_button_placeholder.button("🔍 Analyze Emotion", key="analyze_btn")

        while True:
            ret, frame = camera.read()
            if not ret:
                st.warning("Failed to access the camera.")
                break

            frame = cv2.flip(frame, 1)
            st.session_state.last_frame = frame

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            stframe.image(img, channels="RGB", use_container_width=True)

            # Handle analyze button press
            if analyze:
                st.session_state.analyze_triggered = True
                break

            time.sleep(0.03)

            # Stop camera when checkbox unchecked
            if not st.session_state.get("run_checkbox_value", True):
                break

        if st.session_state.analyze_triggered:
            captured_image = st.session_state.last_frame
            rgb_img = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            name = f"live_caption_{uuid.uuid4()}"

            # Simulate prediction
            emotion, confidence = predict_emotion(captured_image, model, processor)

            st.success("Image captured successfully! Restart camera to capture again")
            result_placeholder.markdown(f"### Emotion: **{emotion}** {emotions_emoji[emotion.lower()]} ({confidence*100:.1f}%)")
            save_values_to_table(name, emotion, confidence, image_bytes)

        # Release camera when done
        camera.release()
        st.session_state.camera = None

    else:
        if st.session_state.camera:
            st.session_state.camera.release()
            st.session_state.camera = None
        stframe.empty()
        analyze_button_placeholder.empty()
        
        
with history_tab:
    st.subheader("Detection History")

    cursor.execute("SELECT id, image_name, detected_emotion, confidence, data, timestamp FROM image_emotion ORDER BY id DESC")
    records = cursor.fetchall()

    if records:
        st.markdown(f"**Total Records:** {len(records)}")

        for id, name, emotion, confidence, data, timestamp in records:
            col1, col2 = st.columns([1, 3])

            with col1:
                image = Image.open(io.BytesIO(data))
                st.image(image, width=100)

            with col2:
                st.write(f"**Image Name:** {name}")
                st.write(f"**Detected Emotion:** {emotion.capitalize()} {emotions_emoji[emotion.lower()]}")
                st.write(f"**Confidence:** {(confidence * 100):.2f}%")
                st.write(f"**{timestamp}**")
                st.button("Delete", key=f"delete_{id}", on_click=delete_values_from_table, args=(id,))
                st.markdown("---")
    else:
        st.info("No history available. Please upload images to see detection history.")