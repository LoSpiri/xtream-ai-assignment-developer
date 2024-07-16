from logging import Logger
import optuna
from sklearn.model_selection import train_test_split

from src.const.model import ModelFactory
from src.utils.load_config import LoadUtils


class OptunaUtils:
    @staticmethod
    def objective(
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
