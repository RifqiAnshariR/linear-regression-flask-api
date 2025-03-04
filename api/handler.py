from flask import request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("../model/wine_quality_model.pkl")

# Handler untuk prediksi
def predict_handler():
    data = request.json["features"]
    data = np.array([data])  # Pastikan bentuknya 2D
    # Konversi ke DataFrame dengan nama kolom yang sama seperti saat training
    df = pd.DataFrame(data, columns=model.feature_names_in_)
    prediction = model.predict(df)
    prediction_rounded = np.rint(prediction).tolist()  # Membulatkan ke nilai terdekat
    return jsonify({"Prediksi": prediction.tolist(),
                    "Prediksi (dibulatkan)":prediction_rounded})
