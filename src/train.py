import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Dataset
def load_data():
    wine_quality = fetch_ucirepo(id=186)
    x = wine_quality.data.features
    # x = wine_quality.data.features[selected_features] # selected_features = ["fixed_acidity", "volatile_acidity", "citric_acid"]
    y = wine_quality.data.targets.values.ravel()  # 1D Array
    return x, y

# Prepare Data
def prepare_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

# Train Model
def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

# Train Pipeline
def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = prepare_data(x, y)

    model = train_model(x_train, y_train)

    joblib.dump(model, "../model/winequality_prediction.pkl")
    joblib.dump(x_test, "../data/x_test.pkl")
    joblib.dump(y_test, "../data/y_test.pkl")

if __name__ == "__main__":
    main()
