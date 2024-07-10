import datetime as dt
from datetime import datetime, timedelta
from io import BytesIO
import json
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
import tarfile
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from typing import Any, Dict, List, Optional, Tuple

from const.path import CONFIG_FOLDER

DEFAULT_CONFIG = CONFIG_FOLDER.joinpath("default.json")


class ConfigParser:

    @staticmethod
    def retrieve_config(file_path: Path) -> dict:
        """
        Retrieve configuration from a JSON file. If the file does not exist, return the default configuration.
        """
        if not file_path.exists():
            file_path = DEFAULT_CONFIG
        else:
            try:
                with open(file_path) as f:
                    configuration = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError("Error decoding JSON") from e
            except PermissionError as e:
                raise PermissionError("Permission denied") from e
        return configuration
    
    @staticmethod
    def get_value(config: dict, key: List[str]) -> Any:
        """
        Retrieve a nested value from the configuration dictionary. If it not available, return an error.
        """
        try:
            for k in key:
                config = config[k]
            return config
        except KeyError as e:
            raise KeyError(f"Key {key} not found in configuration") from e
