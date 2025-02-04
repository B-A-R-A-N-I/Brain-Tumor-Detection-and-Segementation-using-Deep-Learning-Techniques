import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
import joblib
import numpy as np
from preprocessing import preprocess_image
from feature_extraction import extract_features
from tensorflow.keras.applications import VGG16, ResNet101

app = Flask(__name__)

# Load trained model and feature selection mask
model = joblib.load("trained_model.pkl")
selected_mask = joblib.load("selected_features.pkl")
print(f"Loaded feature selection mask with {selected_mask.sum()} features.")

pretrained_models = [VGG16, ResNet101]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        img = preprocess_image(img_path)

        features = extract_features(pretrained_models, np.expand_dims(img, axis=0))
        features_selected = features[:, selected_mask]

        prediction = model.predict(features_selected)
        result = "Tumor" if prediction[0] == 1 else "No Tumor"

        return jsonify({"prediction": result})

    return jsonify({"error": "Invalid file"})


@app.route('/result')
def result():
    prediction = request.args.get("prediction", "Error")
    return render_template("result.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
