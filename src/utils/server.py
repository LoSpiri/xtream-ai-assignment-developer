import numpy as np
import pandas as pd


class ServerUtils:

    @staticmethod
    def parse_payload(payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        # Convert object types to categorical
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')
        return df
