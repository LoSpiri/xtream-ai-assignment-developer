# app.py
from logging import Logger
from pathlib import Path
from flask import Flask, g, request, jsonify

from src.deploy.model_deploy import ModelDeploy


app = Flask(__name__)


@app.route('/health', methods=['GET'])
def home():
    return "Hello, Flask!"


@app.route('/predictprice', methods=['POST'])
def predict_price():
    try:
        payload = request.get_json()
        result = g.model_deployer.predict_price(payload)
    except Exception as e:
        raise e
    return jsonify(result)


@app.route('/similardiamonds', methods=['POST'])
def similar_diamonds():
    try:
        payload = request.get_json()
        result = g.model_deployer.similar_diamonds(payload)
    except Exception as e:
        raise e
    return jsonify(result)


def create_app(model_folder_path: Path, logger: Logger, host='127.0.0.1', port=5000, debug=True):
    model_deployer = ModelDeploy(model_folder_path=model_folder_path, logger=logger)
    model_deployer.run()
    app.config['model_deployer'] = model_deployer

    @app.before_request
    def before_request():
        g.model_deployer = app.config['model_deployer']
    app.run(host=host, port=port, debug=debug, use_reloader=False)
