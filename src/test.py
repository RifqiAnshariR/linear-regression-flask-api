import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate Model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    return y_pred, accuracy, cls_report

# Visualization
def visualize(model, x_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))
    labels = np.unique(y_test)

    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    feature_importance = pd.Series(model.feature_importances_, index=x_test.columns)
    plt.figure(figsize=(10, 6))
    feature_importance.nlargest(10).plot(kind='barh', title="Top 10 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    
    plt.show()

# Test Pipeline
def main():
    model = joblib.load("../model/winequality_prediction.pkl")
    x_test = joblib.load("../data/x_test.pkl")
    y_test = joblib.load("../data/y_test.pkl")
    
    y_pred, accuracy, cls_report = evaluate_model(model, x_test, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", cls_report)
    
    visualize(model, x_test, y_test, y_pred)

if __name__ == "__main__":
    main()
