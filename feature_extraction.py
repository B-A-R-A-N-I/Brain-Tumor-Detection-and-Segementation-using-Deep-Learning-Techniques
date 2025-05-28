import numpy as np

# Function to extract deep features from selected models
def extract_features(models, data):
    feature_list = []
    print("Extracting features from models...")

    for model in models:
        print(f"Extracting features using {model.__name__}...")
        feature_extractor = model(weights='imagenet', include_top=False, pooling='avg')
        features = feature_extractor.predict(data, verbose=1)
        feature_list.append(features)

    print(f"Feature extraction complete: {len(feature_list)} models processed.")

    # Concatenate features from selected models
    concatenated_features = np.concatenate(feature_list, axis=1)
    print(f"Final feature vector shape: {concatenated_features.shape}")
    return concatenated_features

# Example usage:
# models = [VGG16, ResNet101]
# extracted_features = extract_features(models, X)
