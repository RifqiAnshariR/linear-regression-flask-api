from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# 1. Ambil dataset Wine Quality
wine_quality = fetch_ucirepo(id=186)

# 2. Pisahkan fitur (X) dan target (y)
x = wine_quality.data.features
y = wine_quality.data.targets
# x = wine_quality['data']['features']
# y = wine_quality['data']['targets']

# 3. Pisahkan data menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. Inisialisasi dan latih model Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

# Export model
joblib.dump(model, "../model/wine_quality_model.pkl")
