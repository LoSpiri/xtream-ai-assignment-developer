from logging import Logger
from pathlib import Path

from src.const.model import MODELS
from src.model.data_preparation import DataPreparation
from src.model.utils.config import ConfigParser
from src.model.utils.time import time_it


class ModelTrainer:
    def __init__(self, config_file: Path, logger: Logger, model_name: str) -> None:
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger
        data = DataPreparation(
            config_file=config_file,
            logger=self.logger
        )
        self.x_train, self.x_test, self.y_train, self.y_test = data.run()
        self.model_epoch_folder = data.model_epoch_folder

    def run(self) -> None:
        if ConfigParser.get_value(self.configuration, ["model", "enabled"]):
            self.logger.info("Training model...")
            self._train()
            self.logger.info("Model training completed successfully")
        else:
            self.logger.info("Model training is disabled, stopping after data preparation")

    def _train(self) -> None:
        self.model = MODELS[ConfigParser.get_value(self.configuration, ["model", "type"])]
        # TODO: Check it's working
        self.model.fit = time_it(func=self.model.fit, logger=self.logger)
        self.model.fit(self.x_train, self.y_train)
        self.logger.info("Model training completed successfully")
        self._metrics_generation()

    def _metrics_generation(self) -> None:
        self.logger.info("Generating metrics...")

        # self._save_metrics()
        self.logger.info("Metrics generation completed successfully")
