from flask import request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("../model/winequality_prediction.pkl")

# Prediction Handler
def predict_handler():
    data = request.json["features"]
    data = np.array([data])  # 2D Array
    df = pd.DataFrame(data, columns=model.feature_names_in_)
    prediction = model.predict(df)
    return jsonify({"Prediksi": prediction.tolist()})
