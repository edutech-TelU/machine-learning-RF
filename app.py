from flask import Flask, request, jsonify, redirect
from firebase_config import db
import os
import librosa
import numpy as np
import joblib
import uuid
from google.cloud import firestore

app = Flask(__name__)

# API Key
API_KEY = os.getenv("FLASK_API_KEY", "default_api_key")

# URL Model
MODEL_URL = "https://github.com/edutech-TelU/machine-learning-RF/releases/download/v.1.6.0/random_forest_best_model_v1_6_0.pkl"
MODEL_PATH = "random_forest_best_model_v1_6_0.pkl"

# Cek dan unduh model
if not os.path.exists(MODEL_PATH):
    import requests
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as file:
        file.write(response.content)

model = joblib.load(MODEL_PATH)

@app.before_request
def authenticate_request():
    key = request.headers.get('X-API-KEY')
    if key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 403

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_email = request.form.get('email')
        audio_file = request.files.get("audio")

        if not audio_file or not user_email:
            return jsonify({"error": "Audio file and email are required"}), 400

        file_path = f"temp_{uuid.uuid4().hex}.wav"
        audio_file.save(file_path)

        features = process_audio(file_path)
        prediction_proba = model.predict_proba(features)
        prediction = int(prediction_proba[0][1] > 0.65)

        db.collection("users").document(user_email).collection("history").add({
            "result": "Benar" if prediction == 1 else "Salah",
            "confidence": prediction_proba[0].tolist(),
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        os.remove(file_path)

        return jsonify({
            "result": "Benar" if prediction == 1 else "Salah",
            "confidence": prediction_proba[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
