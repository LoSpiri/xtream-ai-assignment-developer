import argparse
from pathlib import Path
from model.trainer import trainer
from src.const.path import CONFIG_FOLDER


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('-c', '--config_file', type=Path,
                        default=CONFIG_FOLDER.joinpath("default.json"),
                        help="Path to the configuration file. Default is 'default.json' in the 'config' folder.")

    args = parser.parse_args()

    # TODO: Add logging
    logger = setup_logger()

    model_trainer = trainer(config_file=args.training_config_file,
                            logger=logger)
    