import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("../model/wine_quality_model.pkl")

# Buat data baru dengan feature names
feature_names = model.feature_names_in_  # Ambil feature names dari model
new_data = pd.DataFrame([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]], columns=feature_names)

# Prediksi
prediction = model.predict(new_data)
print("Prediksi:", prediction)
predicted_quality = np.rint(prediction)  # Membulatkan ke nilai terdekat
print("Prediksi (dibulatkan):", predicted_quality)
