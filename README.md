# Flask API server for Machine Learning predictions using Linear Regression

---
## Run using
    ```bash
    curl -X POST "http://127.0.0.1:5001/api/predict" -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}"