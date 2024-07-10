import time
import io
from logging import Logger
from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import requests

from const.path import CONFIG_FOLDER, TRAIN_FOLDER
from model.utils.config import ConfigParser

#########################################


class ModelTrainer:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger

    def train(self) -> None:
        self.logger.info("Initializing model training...")
        self._setup()
        dataset = self._get_dataset()
        x_train, x_test, y_train, y_test = self._data_preparation(dataset=dataset)
        # TODO: Add model training
        self.logger.info("Model trained successfully")

    def _setup(self) -> None:
        current_unix_epoch = int(time.time())
        model_name = ConfigParser.get_value(self.configuration, ["data", "name"])
        model_id = f"{model_name}_{current_unix_epoch}"
        self.model_epoch_folder = TRAIN_FOLDER.joinpath(model_name).joinpath(str(current_unix_epoch))
        self.logger.info(f"Model ID: {model_id}")
        self.model_epoch_folder.mkdir(parents=True, exist_ok=True)

    def _get_dataset(self) -> pd.DataFrame:
        getLocal = ConfigParser.get_value(
            self.configuration, ["data", "source", "getLocal"]
        )
        if getLocal:
            source_path = ConfigParser.get_value(
                self.configuration, ["data", "source", "localPath"]
            )
        else:
            source_path = ConfigParser.get_value(
                self.configuration, ["data", "source", "urlPath"]
            )
        self.logger.info(f"Selected config file is: '{source_path}'")
        if getLocal:
            data = self._load_path(source_path)
        else:
            data = self._load_url(source_path)
        return data

    def _load_url(self, url: str) -> pd.DataFrame:
        try:
            self.logger.info(f"Fetching data from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = pd.read_csv(io.BytesIO(response.content))
            self.logger.info("Data fetched and read into DataFrame successfully")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for URL: {url}. Got error: {e}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"No data found at URL: {url}")
            raise
        except pd.errors.ParserError:
            self.logger.error(f"Error parsing data from URL: {url}")
            raise
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while fetching data from URL: {url}. Error: {e}"
            )
            raise
        return data

    def _load_path(self, path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
        except Exception as e:
            self.logger.error(f"Failed to read data from path: {path}. Got error: {e}")
            raise
        return data

    def _data_preparation(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info("Preparing data for training...")
        dataset = self._data_cleaning(dataset=dataset)
        self._data_exploration(dataset=dataset)


        target = ConfigParser.get_value(self.configuration, ["data", "target"])
        features = ConfigParser.get_value(self.configuration, ["data", "features"])
        x = dataset[features]
        y = dataset[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        self.logger.info("Data prepared successfully")
        return x_train, x_test, y_train, y_test
    
    def _data_cleaning(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning data...")
        # Drop duplicates
        if ConfigParser.get_value(self.configuration, ["data", "clean", "dropDuplicates"]):
            dataset.drop_duplicates(inplace=True)
        # Drop missing values
        if ConfigParser.get_value(self.configuration, ["data", "clean", "dropNa"]):
            dataset.dropna(inplace=True)
        # Drop columns
        if ConfigParser.get_value(self.configuration, ["data", "clean", "dropColumns", "enabled"]):
            columns = ConfigParser.get_value(self.configuration, ["data", "clean", "dropColumns", "columns"])
            dataset.drop(columns=columns, inplace=True)
        # Drop columns custom condition
        if ConfigParser.get_value(self.configuration, ["data", "clean", "dropColumnsCustom", "enabled"]):
            for column, bounds in ConfigParser.get_value(self.configuration, ["data", "clean", "dropColumnsCustom"]).items():
                min_value = bounds["min"]
                max_value = bounds["max"]
                dataset = dataset[dataset[column] >= min_value]
                dataset = dataset[dataset[column] < max_value]
        self.logger.info("Data cleaned successfully")
        return dataset
    
    def _data_exploration(self, dataset: pd.DataFrame) -> None:
        self.logger.info("Exploring data...")
        figsize = (
            ConfigParser.get_value(self.configuration, ["data", "exploration", "figsize", "width"]),
            ConfigParser.get_value(self.configuration, ["data", "exploration", "figsize", "height"])
        )
        # Get statistics
        if ConfigParser.get_value(self.configuration, ["data", "exploration", "scatter_matrix", "enabled"]):
            scatter_matrix(dataset.select_dtypes(include=['number']), figsize=figsize)
            plt.savefig(self.model_epoch_folder.joinpath("scatter_matrix.png"))
            plt.close()
        if ConfigParser.get_value(self.configuration, ["data", "exploration", "hist", "enabled"]):
            bins = ConfigParser.get_value(self.configuration, ["data", "exploration", "hist", "bins"])
            dataset.hist(bins=bins, figsize=figsize)
            plt.savefig(self.model_epoch_folder.joinpath("hist.png"))
            plt.close()


############################################################################################################


def trainer(
    logger: Logger, config_file: Path = CONFIG_FOLDER.joinpath("default.json")
) -> ModelTrainer:
    model = ModelTrainer(config_file=config_file, logger=logger)
    model.train()

    return model
