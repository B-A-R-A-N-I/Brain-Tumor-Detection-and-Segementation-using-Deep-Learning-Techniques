import joblib
from sklearn.feature_selection import SelectKBest, f_classif


# Function to apply SelectKBest feature selection
def select_best_features(features, labels, k=1000):
    print("Applying SelectKBest for feature selection...")
    selector = SelectKBest(score_func=f_classif, k=k)  # Select top 'k' best features
    selected_features = selector.fit_transform(features, labels)

    # Save the feature selection mask
    selected_mask = selector.get_support()
    joblib.dump(selected_mask, 'selected_features.pkl')
    print(f"Saved feature selection mask with {selected_mask.sum()} features.")

    return selected_features

# Example usage:
# selected_features = select_best_features(features, labels, k=1000)
