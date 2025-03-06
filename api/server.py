from flask import Flask
from flask_cors import CORS
from route import configure_routes

app = Flask(__name__)
CORS(app)

configure_routes(app)

if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5001, debug=True)
