import argparse
import logging
from pathlib import Path
from src.deploy.app import create_app
from src.model.model_trainer import ModelTrainer
from src.const.path import CONFIG_FOLDER
from src.utils.config import ConfigParser


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "-c",
        "--config_file",
        type=Path,
        default="default.json",
        help="Path to the configuration file. Default is 'default.json' in the 'config' folder.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(args.config_file)
    config_file = CONFIG_FOLDER.joinpath(args.config_file)
    configuration = ConfigParser.retrieve_config(config_file)

    if not (ConfigParser.get_value(
        configuration, ["deploy", "enabled"]
    ) and ConfigParser.get_value(
        configuration, ["deploy", "model_name", "trainOnTheSpot"]
    )):
        model_trainer = ModelTrainer(config_file=config_file, logger=logger)
        model_trainer.run()

    create_app(config_file=config_file, logger=logger, debug=True)


if __name__ == "__main__":
    main()
