from pathlib import Path
from logging import Logger

import joblib
import numpy as np

from src.const.path import TRAIN_FOLDER
from src.model.model_trainer import ModelTrainer
from src.utils.config import ConfigParser
from src.utils.server import ServerUtils


class ModelDeploy:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.config_file = config_file
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger

    def run(self) -> None:
        self.logger.info("Deploying model...")
        self.data_processor = ModelTrainer(
            config_file=self.config_file, logger=self.logger
        )
        if ConfigParser.get_value(
            self.configuration, ["deploy", "model_name", "trainOnTheSpot"]
        ):
            self.data_processor.run()
            self.model = joblib.load(
                self.data_processor.model_epoch_folder.joinpath("model.pkl")
            )
        else:
            self.model = joblib.load(
                TRAIN_FOLDER.joinpath(
                    ConfigParser.get_value(self.configuration, ["data", "name"])
                )
                .joinpath(
                    ConfigParser.get_value(
                        self.configuration, ["deploy", "model_name", "epoch"]
                    )
                )
                .joinpath("model.pkl")
            )
        self.logger.info("Model deployed successfully")

    def predict_price(self, payload: dict) -> dict:
        self.logger.info(f"Making prediction with data: {payload}")
        data = ServerUtils.parse_payload(payload)
        target = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "target"]
        )
        data[target] = 42
        data_processed = self.data_processor.data.data_preparation(
            dataset=data, exploration=False
        )
        self.logger.info(data_processed)
        data_processed.drop(columns=target, inplace=True)
        prediction = self.model.predict(data_processed)
        prediction = self.data_processor.inverse_transformation(prediction)
        self.logger.info(f"Prediction: {prediction}")
        payload.update({"prediction": prediction.tolist()})
        return payload

    def similar_diamonds(self, payload: dict) -> dict:
        self.logger.info(f"Generating similar diamonds with data: {payload}")
        data = payload
        dataset = self.data_processor.data.get_dataset()
        dataset = dataset[
            (dataset["cut"].str.strip().str.lower() == data["cut"].strip().lower())
            & (
                dataset["color"].str.strip().str.lower()
                == data["color"].strip().lower()
            )
            & (
                dataset["clarity"].str.strip().str.lower()
                == data["clarity"].strip().lower()
            )
        ]
        dataset["similarity"] = np.abs(dataset["carat"] - data["carat"])
        number_of_similar_diamonds = payload.get("n", 5)
        return (
            dataset.sort_values(by="similarity")
            .head(number_of_similar_diamonds)
            .to_dict()
        )
