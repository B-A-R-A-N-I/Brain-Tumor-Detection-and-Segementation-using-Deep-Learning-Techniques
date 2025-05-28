import joblib
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet101, DenseNet201
from feature_extraction import extract_features
from preprocessing import load_and_preprocess_data
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model and selected features mask
voting_clf = joblib.load('trained_model.pkl')
selected_features = joblib.load('selected_features.pkl')

# Ensure selected_features is a boolean mask
selected_features = selected_features.astype(bool)
print(f"Loaded feature selection mask with {selected_features.sum()} selected features.")

# Load and preprocess test dataset
print("Loading and preprocessing test dataset...")
X_test, y_test = load_and_preprocess_data('dataset2')
print(f"Test data loaded: {X_test.shape[0]} samples")

# Extract features using the same models as training
print("Extracting features for test dataset...")
test_features = extract_features([VGG16, InceptionV3, ResNet101, DenseNet201], X_test)
print(f"Test features extracted: {test_features.shape}")

# Apply feature selection
if selected_features.shape[0] != test_features.shape[1]:
    raise ValueError(f"Feature count mismatch! Expected {selected_features.shape[0]}, got {test_features.shape[1]}.")

X_test_selected = test_features[:, selected_features]
print(f"Selected test feature count: {X_test_selected.shape[1]}")

# Classify using the trained model
print("Predicting test data...")
y_pred = voting_clf.predict(X_test_selected)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Voting Classifier")
plt.show()