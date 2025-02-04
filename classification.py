import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Function to train and select the best classifier
def train_best_classifier(X_train, X_test, y_train, y_test):
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='linear', probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss')
    }

    best_model = None
    best_accuracy = 0

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {acc * 100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = clf

    print(f"Best Classifier: {best_model.__class__.__name__} with {best_accuracy * 100:.2f}% accuracy")
    joblib.dump(best_model, 'trained_model.pkl')
    return best_model


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Tumor", "Tumor"],
                yticklabels=["No Tumor", "Tumor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Example usage:
# best_model = train_best_classifier(X_train, X_test, y_train, y_test)
