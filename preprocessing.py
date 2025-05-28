import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Function to apply cropping and filtering
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    # Crop unnecessary black areas (assuming brain MRIs have black backgrounds)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        if w > 100 and h > 100:  # Prevents over-cropping
            img = img[y:y + h, x:x + w]

    # Resize the image to 224x224 (model requirement)
    img = cv2.resize(img, (224, 224))

    # Apply median filtering to reduce noise (less filtering)
    img = cv2.medianBlur(img, 1)

    # Convert to RGB for deep models
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return preprocess_input(img_to_array(img))


# Function to load and preprocess dataset
def load_and_preprocess_data(dataset_dir='dataset'):
    data = []  # List to store images
    labels = []  # List to store corresponding labels

    tumor_dir = os.path.join(dataset_dir, 'Tumor')
    no_tumor_dir = os.path.join(dataset_dir, 'No_Tumor')

    # Load images from Tumor folder
    for filename in os.listdir(tumor_dir):
        img_path = os.path.join(tumor_dir, filename)
        img = preprocess_image(img_path)
        data.append(img)
        labels.append(1)  # Label 1 for Tumor

    # Load images from No_Tumor folder
    for filename in os.listdir(no_tumor_dir):
        img_path = os.path.join(no_tumor_dir, filename)
        img = preprocess_image(img_path)
        data.append(img)
        labels.append(0)  # Label 0 for No Tumor

    data = np.array(data)
    labels = np.array(labels)

    # Apply data augmentation
    datagen.fit(data)

    return data, labels

# Example usage:
# X, y = load_and_preprocess_data()  # Call this function to load your data
