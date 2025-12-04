# api.py
from flask import Flask
from past_api import past_bp
from future_api import future_bp

app = Flask(__name__)
app.register_blueprint(past_bp)
app.register_blueprint(future_bp)

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
