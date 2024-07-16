# app.py
from logging import Logger
from pathlib import Path
from flask import Flask, g, request, jsonify
from pydantic import ValidationError

from src.const.path import DB_PATH
from src.deploy.database import InteractionDatabase
from src.deploy.model_deploy import ModelDeploy
from src.utils.request_body import PredictPricePayload, SimilarDiamondsPayload


app = Flask(__name__)
interaction_db = InteractionDatabase(db_path=DB_PATH)


@app.route('/health', methods=['GET'])
def home():
    return "Hello, Flask!"


@app.route('/predictprice', methods=['POST'])
def predict_price():
    try:
        payload = request.get_json()
        validated_payload = PredictPricePayload(**payload)
        result = g.model_deployer.predict_price(validated_payload.dict())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


@app.route('/similardiamonds', methods=['POST'])
def similar_diamonds():
    try:
        payload = request.get_json()
        validated_payload = SimilarDiamondsPayload(**payload)
        result = g.model_deployer.similar_diamonds(validated_payload.dict())
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)

@app.route('/interactions', methods=['GET'])
def get_interactions():
    interactions = interaction_db.get_interactions()
    return jsonify(interactions)


def create_app(config_file: Path, logger: Logger, host='127.0.0.1', port=5000, debug=True):
    model_deployer = ModelDeploy(config_file=config_file, logger=logger)
    model_deployer.run()
    app.config['model_deployer'] = model_deployer

    @app.before_request
    def before_request():
        g.model_deployer = app.config['model_deployer']

    @app.after_request
    def log_request_response(response):
        interaction_db.log_interaction(request, response)
        return response

    app.run(host=host, port=port, debug=debug, use_reloader=False)
