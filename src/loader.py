from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff

from src.enums import Modes
from src.ml_common import Dataset


# arff.loadarff() carrega strings em bytes, esta função serve para transformá-las em strings
def load_arff(filename: str) -> tuple[pd.DataFrame, Any]:
    data, metadata = arff.loadarff(filename)
    frame = pd.DataFrame(data)

    for column in frame.select_dtypes([object]).columns:
        frame[column] = frame[column].apply(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )

    return frame, metadata


def load_dataset(filename: str, mode: Modes) -> Dataset:
    start = perf_counter()
    frame, meta = load_arff(filename)

    if not isinstance(mode, Modes):
        mode = Modes(mode)

    match mode:
        case Modes.CLASSIFICATION:
            features, target, target_names = load_classification_dataset(frame, meta)
        case Modes.REGRESSION:
            features, target, target_names = load_regression_dataset(frame)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    load_time = perf_counter() - start
    return Dataset(
        name=Path(filename).name,
        features=features,
        target=target,
        target_names=target_names,
        load_time=load_time,
        feature_names=[str(column) for column in frame.columns[:-1]],
        target_name=str(frame.columns[-1]),
    )


def load_classification_dataset(frame, meta) -> Dataset:
    target_column = frame.columns[-1]
    target_values = frame[target_column]
    _validate_classification_target(target_values)

    features = _feature_matrix(frame, target_column, "Classification")

    target = np.array(
        [normalize_target_value(value) for value in target_values],
        dtype=str,
    )
    target_names = select_target_names(
        target,
        extract_arff_target_names(meta),
    )

    return features, target, target_names


def load_regression_dataset(frame) -> Dataset:
    target_column = frame.columns[-1]
    target_values = frame[target_column]
    _validate_regression_target(target_values)

    features = _feature_matrix(frame, target_column, "Regression")

    try:
        target = target_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Regression target must be numeric.") from exc

    return features, target, None


def normalize_target_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def extract_arff_target_names(meta) -> list[str] | None:
    if meta is None:
        return None

    target_attribute = meta.names()[-1]
    kind, labels = meta[target_attribute]

    if kind != "nominal" or not labels:
        return None

    return [normalize_target_value(label) for label in labels]


def select_target_names(values, declared_names=None) -> list[str]:
    present = set(values)

    if declared_names is not None:
        selected = [name for name in declared_names if name in present]
        if present <= set(selected):
            return selected

    return sorted(present)


def _validate_classification_target(values):
    if values.isna().any():
        raise ValueError("Classification target contains missing values.")

    normalized = values.map(normalize_target_value)
    if normalized.nunique() < 2:
        raise ValueError("Classification target must contain at least two classes.")


def _validate_regression_target(values):
    if values.isna().any():
        raise ValueError("Regression target contains missing values.")

    if values.empty:
        raise ValueError("Regression target must not be empty.")


def _feature_matrix(frame, target_column: str, mode_name: str) -> np.ndarray:
    feature_frame = frame.drop(columns=target_column)

    try:
        features = feature_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{mode_name} features must be numeric or binary numeric columns."
        ) from exc

    if features.size == 0:
        raise ValueError(f"{mode_name} features must not be empty.")

    return features
