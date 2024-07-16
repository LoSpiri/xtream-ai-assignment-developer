from pathlib import Path
from logging import Logger

from src.const.path import TRAIN_FOLDER
from src.utils.config import ConfigParser
from src.deploy.app import create_app


class WebServer:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.config_file = config_file
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger

    def run(self) -> None:
        if ConfigParser.get_value(self.configuration, ["deploy", "enabled"]):
            self.logger.info("Initializing web server...")
            create_app(config_file=self.config_file, logger=self.logger, debug=True)
