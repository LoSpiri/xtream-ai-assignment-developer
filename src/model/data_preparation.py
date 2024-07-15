import time
from logging import Logger
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.const.path import DATA_FOLDER, TRAIN_FOLDER
from src.model.utils.config import ConfigParser
from src.model.utils.exploration import ExplorationUtils
from src.model.utils.load_config import LoadUtils


class DataPreparation:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info("Initializing data preparation...")
        self._setup()
        dataset = self._get_dataset()
        x_train, x_test, y_train, y_test = self._data_preparation(dataset=dataset)
        self.logger.info("Data preparation completed successfully")
        return x_train, x_test, y_train, y_test

    def _setup(self) -> None:
        # TODO: remove this line
        self.logger.info(self.configuration)

        current_unix_epoch = int(time.time())
        model_name = ConfigParser.get_value(self.configuration, ["data", "name"])
        model_id = f"{model_name}_{current_unix_epoch}"
        self.model_epoch_folder = TRAIN_FOLDER.joinpath(model_name).joinpath(
            str(current_unix_epoch)
        )
        self.logger.info(f"Model ID: {model_id}")
        self.model_epoch_folder.mkdir(parents=True, exist_ok=True)

    def _get_dataset(self) -> pd.DataFrame:
        getLocal = ConfigParser.get_value(
            config=self.configuration, key=["data", "source", "getLocal"]
        )
        if getLocal:
            source_path = ConfigParser.get_value(
                self.configuration, ["data", "source", "localPath"]
            )
            source_path = DATA_FOLDER.joinpath(source_path)
        else:
            source_path = ConfigParser.get_value(
                config=self.configuration, key=["data", "source", "url"]
            )
        self.logger.info(f"Selected config file is: '{source_path}'")
        if getLocal:
            data = LoadUtils.load_path(path=source_path, logger=self.logger)
        else:
            data = LoadUtils.load_url(url=str(source_path), logger=self.logger)
        return data

    def _data_preparation(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info("Preparing data for training...")
        dataset = self._data_cleaning(dataset=dataset)
        self.logger.info(dataset.head())
        self._data_exploration(dataset=dataset)
        return self._data_processing(dataset=dataset)

    def _data_cleaning(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning data...")
        # Drop duplicates
        if ConfigParser.get_value(
            self.configuration, ["data", "cleaning", "dropDuplicates"]
        ):
            dataset.drop_duplicates(inplace=True)
        # Drop missing values
        if ConfigParser.get_value(self.configuration, ["data", "cleaning", "dropNa"]):
            dataset.dropna(inplace=True)
        # Drop columns
        if ConfigParser.get_value(
            self.configuration, ["data", "cleaning", "dropColumns", "enabled"]
        ):
            columns = ConfigParser.get_value(
                self.configuration, ["data", "cleaning", "dropColumns", "columns"]
            )
            dataset.drop(columns=columns, inplace=True, errors="ignore")
        # Drop columns custom condition
        if ConfigParser.get_value(
            self.configuration, ["data", "cleaning", "dropCustom", "enabled"]
        ):
            for column, bounds in ConfigParser.get_value(
                self.configuration, ["data", "cleaning", "dropCustom", "columns"]
            ).items():
                if bounds["rangeOrEqual"]:
                    min_value = bounds["min"]
                    max_value = bounds["max"]
                    dataset = dataset[dataset[column] >= min_value]
                    dataset = dataset[dataset[column] < max_value]
                else:
                    value = bounds["value"]
                    dataset = dataset[dataset[column] != value]
        self.logger.info("Data cleaned successfully")
        return dataset

    def _data_exploration(self, dataset: pd.DataFrame) -> None:
        self.logger.info("Exploring data...")
        # TODO: Spostare gli accessi alle configurazioni nei metodi di ExplorationUtils
        # TODO: Mettere i tipi di plot in una const come metric e model e iterare
        figsize = (
            ConfigParser.get_value(
                self.configuration, ["data", "exploration", "figsize", "width"]
            ),
            ConfigParser.get_value(
                self.configuration, ["data", "exploration", "figsize", "height"]
            ),
        )
        # Get statistics
        if ConfigParser.get_value(
            self.configuration, ["data", "exploration", "scatter_matrix", "enabled"]
        ):
            ExplorationUtils.scatter_matrix_plot(
                df=dataset,
                figsize=figsize,
                path=self.model_epoch_folder.joinpath("scatter_matrix.png"),
            )
        if ConfigParser.get_value(
            self.configuration, ["data", "exploration", "hist", "enabled"]
        ):
            bins = ConfigParser.get_value(
                self.configuration, ["data", "exploration", "hist", "bins"]
            )
            ExplorationUtils.hist_plot(
                df=dataset,
                bins=bins,
                figsize=figsize,
                path=self.model_epoch_folder.joinpath("hist.png"),
            )
        if ConfigParser.get_value(
            self.configuration, ["data", "exploration", "categorical", "enabled"]
        ):
            for column in ConfigParser.get_value(
                self.configuration, ["data", "exploration", "categorical", "columns"]
            ):
                ExplorationUtils.violin_plot_by_price(
                    df=dataset,
                    column=column,
                    path=self.model_epoch_folder.joinpath(f"violin_{column}.png"),
                )
                ExplorationUtils.scatter_plot_by_price_vs_carat(
                    df=dataset,
                    column=column,
                    path=self.model_epoch_folder.joinpath(f"scatter_plot_{column}.png"),
                )
        self.logger.info("Data explored successfully")

    def _data_processing(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info("Processing data...")
        # Drop columns
        if ConfigParser.get_value(
            self.configuration,
            ["data", "processing", "dropColumnsPostExploration", "enabled"],
        ):
            columns = ConfigParser.get_value(
                self.configuration,
                ["data", "processing", "dropColumnsPostExploration", "columns"],
            )
            dataset.drop(columns=columns, inplace=True, errors="ignore")
        # Get dummies
        if ConfigParser.get_value(
            self.configuration, ["data", "processing", "getDummies", "enabled"]
        ):
            columns = ConfigParser.get_value(
                self.configuration, ["data", "processing", "getDummies", "columns"]
            )
            for column in columns:
                if not (
                    ConfigParser.get_value(
                        self.configuration,
                        ["data", "processing", "orderCategorical", "enabled"],
                    )
                    and column
                    in ConfigParser.get_value(
                        self.configuration,
                        ["data", "processing", "orderCategorical", "columns"],
                    )
                ):
                    dataset = pd.get_dummies(dataset, columns=[column], drop_first=True)
        # Order categorical attibutes
        if ConfigParser.get_value(
            self.configuration, ["data", "processing", "orderCategorical", "enabled"]
        ):
            for column, categories in ConfigParser.get_value(
                self.configuration,
                ["data", "processing", "orderCategorical", "columns"],
            ).items():
                if not (
                    ConfigParser.get_value(
                        self.configuration,
                        ["data", "processing", "getDummies", "enabled"],
                    )
                    and column
                    in ConfigParser.get_value(
                        self.configuration,
                        ["data", "processing", "getDummies", "columns"],
                    )
                ):
                    dataset[column] = pd.Categorical(
                        dataset[column], categories=categories, ordered=True
                    )
        # Split data
        target = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "target"]
        )
        x = dataset.drop(columns=target)
        y = dataset[target]
        test_size = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "testSize"]
        )
        random_state = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "randomState"]
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.logger.info("Data processed successfully")
        return x_train, x_test, y_train, y_test
