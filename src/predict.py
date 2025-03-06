import joblib
import numpy as np
import pandas as pd

# Load Model
model = joblib.load("../model/winequality_prediction.pkl")

# Test Data
feature_names = model.feature_names_in_
new_data = pd.DataFrame([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]], columns=feature_names)

# Prediction
prediction = model.predict(new_data)
print("Prediksi:", prediction)
