import functools
from logging import Logger
from pathlib import Path
import time

import joblib
import optuna

from src.model.utils.optuna_objective import OptunaUtils
from src.const.transformation import TRANSFORMATIONS
from src.model.utils.exploration import ExplorationUtils
from src.const.metric import METRICS
from src.const.model import ModelFactory
from src.model.data_preparation import DataPreparation
from src.model.utils.config import ConfigParser


class ModelTrainer:
    def __init__(self, config_file: Path, logger: Logger) -> None:
        self.configuration = ConfigParser.retrieve_config(config_file)
        self.logger = logger
        data = DataPreparation(config_file=config_file, logger=self.logger)
        self.x_train, self.x_test, self.y_train, self.y_test = data.run()
        self.model_epoch_folder = data.model_epoch_folder
        self.logger.info(self.x_train.head())
        self.logger.info(self.x_test.head())
        self.logger.info(self.y_train.head())
        self.logger.info(self.y_test.head())
        self.logger.info(self.x_train.info())
        self.logger.info(self.x_test.info())
        self.logger.info(self.y_train.info())
        self.logger.info(self.y_test.info())

    def run(self) -> None:
        if ConfigParser.get_value(self.configuration, ["model", "enabled"]):
            self.logger.info("Training model...")
            self._transformation()
            self._train()
            self._inverse_transformation()
            self._metrics_generation()
            self._model_exploration()
            self._save_model()
            self.logger.info("Model training completed successfully")
        else:
            self.logger.info(
                "Model training is disabled, stopping after data preparation"
            )

    def _train(self) -> None:
        model_name = ConfigParser.get_value(self.configuration, ["model", "type"])
        model_params = ConfigParser.get_value(
            self.configuration, ["model", "parameters"]
        )
        if ConfigParser.get_value(
            self.configuration, ["model", "optuna_tuning", "enabled"]
        ):
            model_best_params = self._tuning(model_name, model_params)
        self.model = ModelFactory.create_model(model_name, **model_best_params)
        start_time = time.time()
        self.model.fit(self.x_train, self.y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Time elapsed for model training: {elapsed_time} seconds")
        self.pred = self.model.predict(self.x_test)

    def _metrics_generation(self) -> None:
        self.logger.info("Generating metrics...")
        if ConfigParser.get_value(
            self.configuration, ["model", "evaluation", "enabled"]
        ):
            metrics = ConfigParser.get_value(
                self.configuration, ["model", "evaluation", "metrics"]
            )
            for metric in metrics:
                if metric not in METRICS:
                    self.logger.error(f"Metric {metric} not found in METRICS constant")
                else:
                    value = round(METRICS[metric](self.y_test, self.pred), 4)
                    self.logger.info(f"{metric}: {value}")
        # TODO: self._save_metrics()
        self.logger.info("Metrics generation completed successfully")

    def _save_model(self) -> None:
        if ConfigParser.get_value(self.configuration, ["model", "save", "enabled"]):
            self.logger.info("Saving model...")
            model_path = self.model_epoch_folder.joinpath(
                ConfigParser.get_value(
                    self.configuration, ["model", "save", "filename"]
                )
            )
            joblib.dump(self.model, model_path)
            self.logger.info("Model saved successfully")

    def _model_exploration(self) -> None:
        if ConfigParser.get_value(
            self.configuration, ["model", "exploration", "gof", "enabled"]
        ):
            self.logger.info("Exploring model...")
            ExplorationUtils.plot_gof(
                self.y_test, self.pred, self.model_epoch_folder.joinpath("gof.png")
            )

    def _transformation(self) -> None:
        if ConfigParser.get_value(
            self.configuration, ["model", "transformation", "enabled"]
        ):
            self.logger.info("Transforming data...")
            for transformation in ConfigParser.get_value(
                self.configuration, ["model", "transformation", "func"]
            ):
                if transformation not in TRANSFORMATIONS:
                    self.logger.error(
                        f"Transformation {transformation} not found in TRANSFORMATIONS constant"
                    )
                else:
                    self.y_train = TRANSFORMATIONS[transformation]["func"](self.y_train)

    def _inverse_transformation(self) -> None:
        if ConfigParser.get_value(
            self.configuration, ["model", "transformation", "enabled"]
        ):
            self.logger.info("Inverse transforming data...")
            for transformation in ConfigParser.get_value(
                self.configuration, ["model", "transformation", "func"]
            ):
                if transformation not in TRANSFORMATIONS:
                    self.logger.error(
                        f"Transformation {transformation} not found in TRANSFORMATIONS constant"
                    )
                else:
                    self.pred = TRANSFORMATIONS[transformation]["inverse_func"](
                        self.pred
                    )

    def _tuning(self, model_name: str, model_params: dict) -> dict:
        self.logger.info("Tuning model...")
        hyperparams = ConfigParser.get_value(
            self.configuration, ["model", "optuna_tuning", "hyperparameters"]
        )
        test_size = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "testSize"]
        )
        random_state = ConfigParser.get_value(
            self.configuration, ["data", "processing", "trainTestSplit", "randomState"]
        )
        metric = METRICS[
            ConfigParser.get_value(
                self.configuration, ["model", "optuna_tuning", "metric"]
            )
        ]
        objective_with_data = functools.partial(
            OptunaUtils.objective,
            x_train_og=self.x_train,
            y_train_og=self.y_train,
            test_size=test_size,
            random_state=random_state,
            metric=metric,
            model_name=model_name,
            model_params=model_params,
            hyperparams=hyperparams,
            logger=self.logger,
        )
        self.logger.info("LS DEBUG: Starting Optuna study...")
        study = optuna.create_study(
            direction=ConfigParser.get_value(
                self.configuration, ["model", "optuna_tuning", "direction"]
            ),
            study_name=model_name
        )
        study.optimize(
            objective_with_data,
            n_trials=ConfigParser.get_value(
                self.configuration, ["model", "optuna_tuning", "nTrials"]
            ),
        )
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters found: {best_params}")
        return best_params
    
    def _objective(
        trial: optuna.trial.Trial,
        x_train_og,
        y_train_og,
        test_size,
        random_state,
        metric,
        model_name,
        model_params,
        hyperparams,
        logger: Logger,
    ) -> float:
        params = LoadUtils.load_hyperparams(
            trial=trial,
            model_name=model_name,
            model_constant_params=model_params,
            hyperparams=hyperparams,
            logger=logger,
        )
        logger.info(x_train_og.info())
        logger.info(y_train_og.info())
        y_train_og["price"] = y_train_og["price"].astype("int")
        x_train, y_train, x_test, y_test = train_test_split(
            x_train_og, y_train_og, test_size=test_size, random_state=random_state
        )
        model = ModelFactory.create_model(model_name, **params)
        logger.info(f"Training model with hyperparameters: {model.get_params()}")
        model.set_params(enable_categorical=True)
        logger.info(x_train.info())
        logger.info(x_test.info())
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        evaluation = metric(y_test, pred)
        logger.info(f"Evaluation for hyperparameters: {evaluation}")
        return evaluation
