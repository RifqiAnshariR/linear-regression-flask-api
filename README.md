# 1. Aktifkan venv (Windows)
venv\Scripts\activate

# 2. Set Flask ke mode development (hanya untuk sesi ini)
set FLASK_ENV=development

# 3. Jalankan Flask (auto-reload aktif)
flask run

---
# Run using
    ```bash
    curl -X POST "http://127.0.0.1:5001/api/predict" -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}"