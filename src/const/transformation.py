import numpy as np


TRANSFORMATIONS = {
    "log": {
        "func": lambda x: np.log(x),
        "inverse_func": lambda x: np.exp(x),
    }
}
