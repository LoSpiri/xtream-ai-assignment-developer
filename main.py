import argparse
import logging
from pathlib import Path
from src.model.model_trainer import ModelTrainer
from src.const.path import CONFIG_FOLDER


if __name__ == "__main__":
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
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
        handlers=[
            logging.StreamHandler()  # Output logs to the console
        ]
    )
    logger = logging.getLogger(__name__)

    # TODO: Add logging
    # logger = Logger(name="Model Training")

    logger.info(args.config_file)
    config_file = CONFIG_FOLDER.joinpath(args.config_file)
    # Check model_name here what it does
    model_trainer = ModelTrainer(
        config_file=config_file, logger=logger
    )
    model_trainer.run()
    logger.info("Model training completed successfully")
