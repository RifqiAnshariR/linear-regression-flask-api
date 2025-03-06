# 📌 Description  
**Flask API server for Machine Learning predictions using Linear Regression**  
This project implements a simple **Linear Regression** model to predict **bike rentals (Y)** based on environmental conditions: **temperature, humidity, and windspeed (X)**.

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
