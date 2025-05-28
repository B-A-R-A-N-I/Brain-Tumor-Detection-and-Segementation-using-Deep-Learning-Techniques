# Brain-Tumor-Detection-and-Segementation-using-Deep-Learning-Techniques
Abstract:
This project presents an AI-based system for the automated detection and segmentation of brain tumors from MRI images. The objective is to assist medical professionals by improving diagnostic accuracy and reducing analysis time. The system comprises two major components: a classification model using Support Vector Machine (SVM) to detect the presence of a tumor, and a U-Net-based Convolutional Neural Network (CNN) for segmenting the tumor region from MRI scans. The dataset is preprocessed using grayscale conversion, normalization, and noise reduction techniques. The SVM model is trained on extracted features to classify tumor vs. non-tumor images, while the U-Net architecture accurately highlights tumor boundaries for further analysis. The project offers a reliable and efficient solution for early diagnosis and medical research support.

# Libraries & Packages:

### Data Handling & Preprocessing:
- numpy – Numerical operations
- pandas – Data management and analysis
- opencv-python – Image loading, preprocessing, and transformation
- matplotlib – Visualization of MRI images and segmentation results
- skimage – Image processing utilities

### Feature Extraction & Classification:
- scikit-learn – SVM model, feature scaling, accuracy metrics, train-test split

### Segmentation (Deep Learning):
- tensorflow / keras – U-Net architecture implementation, training, and evaluation
- PIL – Image handling during preprocessing or augmentation

### Evaluation:
- sklearn.metrics – Accuracy, precision, recall, F1-score, confusion matrix
- tensorflow.keras.metrics – Dice coefficient, IoU (for segmentation evaluation)

