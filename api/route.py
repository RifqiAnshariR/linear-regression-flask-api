from flask import Blueprint
from handler import predict_handler

api_bp = Blueprint("api", __name__)

# Definisikan route
@api_bp.route("/predict", methods=["POST"])
def predict():
    return predict_handler()

@api_bp.route("/", methods=["GET"])
def home():
    return "Flask API is running!"

# Connect Routes to Flask
def configure_routes(app):
    app.register_blueprint(api_bp, url_prefix="/api")
