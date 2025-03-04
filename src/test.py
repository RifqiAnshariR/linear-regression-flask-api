import joblib
from sklearn.metrics import mean_squared_error, r2_score
from train import x_test, y_test

# 5. Import model
model = joblib.load("../model/wine_quality_model.pkl")
y_pred = model.predict(x_test)

# 6. Prediksi
y_pred = model.predict(x_test)

# 7. Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Koefisien: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
