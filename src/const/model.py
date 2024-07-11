from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


MODELS = {
    "linear_regression": LinearRegression,
    "xgb_regression": XGBRegressor,
}
