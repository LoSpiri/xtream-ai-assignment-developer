import io
from pathlib import Path
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
