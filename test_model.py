import joblib
from tensorflow.keras.applications import VGG16, ResNet101
from preprocessing import load_and_preprocess_data
from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained best model
best_model = joblib.load('trained_model.pkl')

# Load the saved feature selection mask
selected_mask = joblib.load('selected_features.pkl')
print(f"Loaded feature selection mask with {selected_mask.sum()} features.")

# Load and preprocess test data
print("Loading and preprocessing test dataset...")
X_test, y_test = load_and_preprocess_data('dataset2')
print(f"Test data loaded: {X_test.shape[0]} samples")

# Extract features using the same models as training
print("Extracting features for test dataset...")
test_features = extract_features([VGG16, ResNet101], X_test)
print(f"Test features extracted: {test_features.shape}")

# Ensure feature selection mask matches test feature size
if selected_mask.shape[0] != test_features.shape[1]:
    raise ValueError(f"Feature count mismatch! Expected {selected_mask.shape[0]}, got {test_features.shape[1]}")

# Apply feature selection
X_test_selected = test_features[:, selected_mask]
print(f"Selected test feature count: {X_test_selected.shape[1]}")

# Classify using the trained model
print("Predicting test data...")
y_pred = best_model.predict(X_test_selected)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Model")
plt.show()
