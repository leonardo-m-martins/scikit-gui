"""Mapping between application modes and their model configurations."""

from src.enums import Modes
from src.ml.classification import CLASSIFICATION_CONFIG
from src.ml.regression import REGRESSION_CONFIG

MODE_CONFIGS = {
    Modes.CLASSIFICATION: CLASSIFICATION_CONFIG,
    Modes.REGRESSION: REGRESSION_CONFIG,
}


def method_values(mode: Modes) -> list[str]:
    return [method.value for method in MODE_CONFIGS[mode].specs]
