from pathlib import Path

import numpy as np

from src.enums import Modes
from src.ml_common import Dataset
from src.shared import ler_arquivo_generico


def load_dataset(filename: str, mode: Modes) -> Dataset:
    if not isinstance(mode, Modes):
        mode = Modes(mode)

    match mode:
        case Modes.CLASSIFICATION:
            return load_classification_dataset(filename)
        case Modes.REGRESSION:
            return load_regression_dataset(filename)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")


def load_classification_dataset(filename: str) -> Dataset:
    frame, meta = ler_arquivo_generico(filename, com_meta=True)

    target_column = frame.columns[-1]
    target_values = frame[target_column]
    _validate_classification_target(target_values)

    features = _feature_matrix(frame, target_column, "Classification")

    target_names = extract_arff_target_names(meta)
    target, target_names = transform_string_targets_into_int(
        target_values,
        target_names,
    )

    return Dataset(
        name=Path(filename).name,
        features=features,
        target=target,
        target_names=target_names,
    )


def load_regression_dataset(filename: str) -> Dataset:
    frame, _ = ler_arquivo_generico(filename, com_meta=True)

    target_column = frame.columns[-1]
    target_values = frame[target_column]
    _validate_regression_target(target_values)

    features = _feature_matrix(frame, target_column, "Regression")

    try:
        target = target_values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Regression target must be numeric.") from exc

    return Dataset(
        name=Path(filename).name,
        features=features,
        target=target,
        target_names=None,
    )


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


def transform_string_targets_into_int(
    values,
    declared_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    normalized_values = [normalize_target_value(value) for value in values]
    present = set(normalized_values)

    if declared_names is not None:
        selected = [name for name in declared_names if name in present]
        if present <= set(selected):
            return _encode_targets(normalized_values, selected)

    names = sorted(present)
    return _encode_targets(normalized_values, names)


def _encode_targets(values, names):
    index = {name: position for position, name in enumerate(names)}
    return np.array([index[value] for value in values], dtype=int), names


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
