# 📌 Description  
**Flask API Server for Machine Learning Predictions using Random Forest Algorithm**  
Flask API server for Machine Learning predictions using Random Forest algorithm. This project implements a **Random Forest Classifier** model to predict **wine quality** (Y) based on various chemical properties: **fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol** (X).

# 🚀 How to Run  
1. **View Dataset Information**  
   ```bash
   python info.py
   
2. **Train the Model**  
   ```bash
   python train.py
   
3. **Test the Model**  
   ```bash
   python test.py

4. **Test the API**
   ```bash
   curl -X POST "http://127.0.0.1:5001/api/predict" -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}"

# ➕ Miscellaneous
1. **UCI Github**
   https://github.com/uci-ml-repo/ucimlrepo
