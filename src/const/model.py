from sklearn.linear_model import LinearRegression
import xgboost


class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "linear_regression":
            return LinearRegression(**kwargs)
        elif model_type == "xgb_regression":
            return xgboost.XGBRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
