from logging import Logger
import optuna
from sklearn.model_selection import train_test_split

from src.const.model import ModelFactory
from src.utils.load_config import LoadUtils

from sklearn.preprocessing import LabelEncoder


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

        # Deep copy original data to avoid modification between trials
        x_train = x_train_og.copy()
        y_train = y_train_og.copy()

        # Encode categorical columns if needed
        categorical_cols = ['cut', 'color', 'clarity']
        label_encoders = {}
        for col in categorical_cols:
            if x_train[col].dtype.name == 'category':
                label_encoders[col] = LabelEncoder()
                x_train[col] = label_encoders[col].fit_transform(x_train[col])

        # Ensure y_train["price"] is of integer type
        y_train["price"] = y_train["price"].astype("int")

        # Split data
        x_train_split, x_test, y_train_split, y_test = train_test_split(
            x_train, y_train, test_size=test_size, random_state=random_state
        )

        # Create the model with the parameters and enable categorical handling if necessary
        model = ModelFactory.create_model(model_name, **params)
        if model_name.startswith('xgboost'):
            params['enable_categorical'] = True

        # Fit the model
        model.fit(x_train_split, y_train_split)
        pred = model.predict(x_test)
        evaluation = metric(y_test, pred)
        logger.info(f"Evaluation for hyperparameters: {evaluation}")

        # Reverse label encoding for categorical columns
        for col, encoder in label_encoders.items():
            x_train[col] = encoder.inverse_transform(x_train[col])

        return evaluation
