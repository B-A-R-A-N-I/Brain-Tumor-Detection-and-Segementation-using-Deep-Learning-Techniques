import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image
from feature_extraction import extract_features

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load trained classifier and feature selection mask
model = joblib.load("trained_model.pkl")
selected_mask = joblib.load("selected_features.pkl")

# Load U-Net segmentation model
unet_model = load_model("unet_brain_tumor.h5")

pretrained_models = [VGG16, ResNet101]

@app.route('/')
def home():
    return render_template('index.html')

def visualize_features(features, save_path):
    """Visualizes and saves extracted features as an image."""
    plt.figure(figsize=(10, 2))
    plt.imshow(features.reshape(1, -1), cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title("Extracted Features Visualization")
    plt.savefig(save_path)
    plt.close()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("result.html", prediction="Error: No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template("result.html", prediction="Error: No selected file")

    if file:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # **Step 1: Original MRI Image**
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (256, 256))
        original_path = os.path.join(PROCESSED_FOLDER, "1_original.jpg")
        cv2.imwrite(original_path, original_img)

        # **Step 2: Convert to Grayscale**
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(PROCESSED_FOLDER, "2_grayscale.jpg")
        cv2.imwrite(gray_path, gray_img)

        # **Step 3: Apply Color Map**
        color_mapped_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        color_mapped_path = os.path.join(PROCESSED_FOLDER, "3_color_map.jpg")
        cv2.imwrite(color_mapped_path, color_mapped_img)

        # **Step 4: Preprocessed Image**
        preprocessed_img = preprocess_image(img_path)
        preprocessed_path = os.path.join(PROCESSED_FOLDER, "4_preprocessed.jpg")
        cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed_img.astype('uint8'), cv2.COLOR_RGB2BGR))

        # **Step 5: Feature Extraction**
        features = extract_features(pretrained_models, np.expand_dims(preprocessed_img, axis=0))
        features_selected = features[:, selected_mask]

        feature_img_path = os.path.join(PROCESSED_FOLDER, "5_feature_extracted.jpg")
        visualize_features(features_selected, feature_img_path)

        # **Step 6: Tumor Detection Image (NEW)**
        prediction = model.predict(features_selected)
        result = "Tumor" if prediction[0] == 1 else "No Tumor"

        detected_path = os.path.join(PROCESSED_FOLDER, "6_predicted_result.jpg")
        detected_img = original_img.copy()

        if result == "Tumor":
            tumor_detected_img = draw_tumor_bounding_box(detected_img, img_path)
            cv2.imwrite(detected_path, tumor_detected_img)
        else:
            cv2.putText(detected_img, "No Tumor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(detected_path, detected_img)

        # **Step 7 & 8: Tumor Segmentation & Mask**
        segmented_path = "None"
        tumor_mask_path = "None"
        if result == "Tumor":
            tumor_detected_path, tumor_mask_path = segment_tumor(img_path, file.filename)

        return redirect(url_for("result", prediction=result,
                                img1="1_original.jpg",
                                img2="2_grayscale.jpg",
                                img3="3_color_map.jpg",
                                img4="4_preprocessed.jpg",
                                img5="5_feature_extracted.jpg",
                                img6="6_predicted_result.jpg",
                                img7="7_tumor_mask.jpg",
                                img8="8_tumor_segmented.jpg"))

def draw_tumor_bounding_box(image, img_path):
    """Draws a red bounding box around the detected tumor region."""
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(original_img, (256, 256))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    mask = unet_model.predict(img_resized)[0]
    mask = (mask > 0.3).astype(np.uint8) * 255
    mask = cv2.resize(mask, (256, 256))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Tumor Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def segment_tumor(img_path, filename):
    """Segments the tumor from the MRI and saves images."""
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(original_img, (256, 256))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    mask = unet_model.predict(img_resized)[0]
    mask = (mask > 0.3).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

    tumor_overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    tumor_overlay = cv2.resize(tumor_overlay, (mask.shape[1], mask.shape[0]))
    tumor_overlay[mask == 255] = [0, 0, 255]

    tumor_detected_path = os.path.join(PROCESSED_FOLDER, "7_tumor_mask.jpg")
    cv2.imwrite(tumor_detected_path, tumor_overlay)

    tumor_mask_path = os.path.join(PROCESSED_FOLDER, "8_tumor_segmented.jpg")
    cv2.imwrite(tumor_mask_path, mask)

    return tumor_detected_path, tumor_mask_path

@app.route('/result')
def result():
    return render_template("result.html", **request.args)

if __name__ == '__main__':
    app.run(debug=True)
