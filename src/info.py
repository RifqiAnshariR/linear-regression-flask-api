from ucimlrepo import fetch_ucirepo

# Ambil dataset Wine Quality
wine_quality = fetch_ucirepo(id=186)

print(wine_quality.data.features.head())  # 5 baris pertama fitur
print(wine_quality.data.targets.head())   # 5 baris pertama target

print(wine_quality.data.features.info())  # Info tipe data

print(wine_quality.data.features.describe())  # Statistik deskriptif

print(wine_quality.data.features.isnull().sum())  # Jumlah missing values per kolom
