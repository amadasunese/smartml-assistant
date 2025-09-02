# flask_agent.py
from flask import Flask, render_template
import importlib
import pkgutil
from routes.views import bp # this is your routes package

def create_app():
    app = Flask(__name__)

    app.register_blueprint(bp)


    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
