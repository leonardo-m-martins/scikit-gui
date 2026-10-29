import numpy as np
from sklearn import metrics as me
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from src.enums import Methods, Preprocessors
from src.ml_common import ModeConfig, ModelSpec
from src.shared import RegressionMetrics

_MODEL_SPECS: dict[Methods, ModelSpec] = {
    Methods.MULTILAYER_PERCEPTRON: ModelSpec(
        MLPRegressor,
        {
            "hidden_layer_sizes": (150,),
            "activation": "relu",
            "solver": "lbfgs",
            "alpha": 0.01,
            "learning_rate": "adaptive",
            "max_iter": 1000,
            "random_state": 42,
        },
        (
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate",
            "max_iter",
            "random_state",
        ),
        preprocessing=Preprocessors.STANDARD_SCALER,
    ),
    Methods.RANDOM_FOREST: ModelSpec(
        RandomForestRegressor,
        {
            "n_estimators": 80,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
        },
        (
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "n_jobs",
        ),
    ),
}


def regression_metrics(y_teste, y_pred, labels=None) -> RegressionMetrics:
    mae = me.mean_absolute_error(y_teste, y_pred)
    mse = me.mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    r2 = me.r2_score(y_teste, y_pred)
    return RegressionMetrics(mae, mse, rmse, r2)


REGRESSION_CONFIG = ModeConfig(_MODEL_SPECS, regression_metrics)
