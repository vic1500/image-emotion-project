# Image Emotion Detection App 😊

A real-time emotion detection application built with Streamlit that can detect emotions from both uploaded images and live camera feed.

Live Demo: [Image Emotion Detection App](https://image-emotion-detection-app.streamlit.app/)

## Features

- 📸 Image upload support for emotion detection
- 🎥 Real-time emotion detection through webcam
- 📊 History tracking of all emotion detections
- 🗃️ Local SQLite database for storing results
- 🎯 High-accuracy emotion detection using ViT (Vision Transformer) model

## Emotions Detected

The application can detect 7 different emotions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## Tech Stack

- Python 3.x
- Streamlit
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- SQLite
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vic1500/image-emotion-project.git
cd image-emotion-project
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

# Project Structure
```code
├── app.py              # Main Streamlit application
├── model.py           # Emotion detection model implementation
├── requirements.txt   # Project dependencies
├── .gitignore        # Git ignore file
└── image_emotion.db  # SQLite database (created on first run)
```

# Model Details
The application uses the `abhilash88/face-emotion-detection model`, which is a Vision Transformer (ViT) based model fine-tuned for emotion detection.

# Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## Author  

**Arowosaye Victor Oluwadamilola (Dami)**  
📚 *Industrial Mathematics (Computer Science option) Student*  
💡 *Aspiring AI/ML Engineer & Data Scientist*  

Passionate about building intelligent systems that merge mathematics, data, and code to solve real-world problems. Experienced in **Python**, **React**, **FastAPI**, and **Machine Learning**, and constantly exploring how AI can empower education and enhance productivity.  

💼 **LinkedIn:** [https://www.linkedin.com/in/victor-arowosaye/]  
🐙 **GitHub:** [https://github.com/vic1500)]  
✉️ **Email:** [victordman15@gmail.com]

