from tensorflow.keras.applications import VGG16, ResNet101
from preprocessing import load_and_preprocess_data
from feature_extraction import extract_features
from feature_selection import select_best_features
from classification import train_best_classifier, plot_confusion_matrix
from sklearn.model_selection import train_test_split

# Load and preprocess data
print("Loading and preprocessing data...")
X, y = load_and_preprocess_data('dataset')
print(f"Data loaded: {X.shape[0]} samples")

# Define pre-trained models
pretrained_models = [VGG16, ResNet101]

# Extract features using pre-trained models
print("Extracting features...")
features = extract_features(pretrained_models, X)
print(f"Features extracted: {features.shape[1]} features")

# Apply feature selection using SelectKBest
print("Applying feature selection...")
selected_features = select_best_features(features, y, k=1000)

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

# Train and select the best classifier
print("Training and selecting the best classifier...")
best_model = train_best_classifier(X_train, X_test, y_train, y_test)

y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred, best_model.__class__.__name__)

print("Training completed successfully! Best model saved as 'trained_model.pkl'.")