from pathlib import Path
from logging import Logger

from src.const.path import TRAIN_FOLDER
from src.utils.config import ConfigParser
from src.deploy.app import create_app


class WebServer:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger

    def run(self) -> None:
        if ConfigParser.get_value(self.configuration, ["deploy", "enabled"]):
            model_name = ConfigParser.get_value(self.configuration, ["data", "name"])
            if ConfigParser.get_value(self.configuration, ["deploy", "model_name", "sameAsTraining"]):
                epoch = sorted(TRAIN_FOLDER.joinpath(model_name).glob("*"))[-1].name
                self.logger.info(f"Model name: {model_name}")
                self.logger.info(f"Epoch: {epoch}")
            else:
                epoch = ConfigParser.get_value(self.configuration, ["deploy", "model_name", "epoch"])
            path_to_folder = TRAIN_FOLDER.joinpath(model_name).joinpath(epoch)
            create_app(model_folder_path=path_to_folder, logger=self.logger, debug=True)
