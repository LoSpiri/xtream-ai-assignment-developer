from pathlib import Path
from logging import Logger

import joblib
import numpy as np

from src.model.model_trainer import ModelTrainer
from src.utils.config import ConfigParser
from src.utils.server import ServerUtils


class ModelDeploy:
    def __init__(self, model_folder_path: Path, logger: Logger) -> None:
        self.model_folder_path = model_folder_path
        self.config_file = model_folder_path.joinpath("config.json")
        self.configuration = ConfigParser.retrieve_config(self.config_file)
        self.logger = logger
        self.model = joblib.load(model_folder_path.joinpath("model.pkl"))
        self.data_processor = ModelTrainer(self.config_file, logger)

    def run(self) -> None:
        pass

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
        return dataset.sort_values(by="Similarity").head(number_of_similar_diamonds).to_dict()
