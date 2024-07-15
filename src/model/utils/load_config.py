import io
from pathlib import Path
import optuna
import pandas as pd
from logging import Logger
import requests


class LoadUtils:
    @staticmethod
    def load_url(url: str, logger: Logger) -> pd.DataFrame:
        try:
            logger.info(f"Fetching data from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = pd.read_csv(io.BytesIO(response.content))
            logger.info("Data fetched and read into DataFrame successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL: {url}. Got error: {e}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"No data found at URL: {url}")
            raise
        except pd.errors.ParserError:
            logger.error(f"Error parsing data from URL: {url}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while fetching data from URL: {url}. Error: {e}"
            )
            raise
        return data

    @staticmethod
    def load_path(path: Path, logger: Logger) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed to read data from path: {path}. Got error: {e}")
            raise
        return data

    @staticmethod
    def load_hyperparams(trial: optuna.trial.Trial, model_name: str, model_constant_params: dict, hyperparams: dict, logger: Logger) -> dict:
        # Define hyperparameters to tune using trial.suggest_* methods directly
        param = {}
        for param_name, param_info in hyperparams.items():
            if 'cat' in param_info:
                param[param_name] = trial.suggest_categorical(param_name, param_info['cat'])
            elif 'range' in param_info:
                if param_info['log']:
                    param[param_name] = trial.suggest_float(param_name, param_info['range'][0], param_info['range'][1], log=True)
                else:
                    if isinstance(param_info['range'][0], int) and isinstance(param_info['range'][1], int):
                        param[param_name] = trial.suggest_int(param_name, param_info['range'][0], param_info['range'][1])
                    else:
                        param[param_name] = trial.suggest_float(param_name, param_info['range'][0], param_info['range'][1])
        param.update(model_constant_params)
        logger.info(f"Final hyperparameters for {model_name}: {param}")
        return param
