import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.shared import MethodResult, Metrics


@dataclass(frozen=True)
class ModelSpec:
    model_type: type[BaseEstimator]
    params: dict[str, Any] = field(default_factory=dict)
    display_keys: tuple[str, ...] = ()
    preprocessing: str | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    features: Any
    target: np.ndarray
    target_names: list[str] | None = None


MetricsFunction = Callable[[np.ndarray, np.ndarray], Metrics]


def get_model_params(
    model: BaseEstimator,
    display_keys: tuple[str, ...] = (),
) -> dict:
    params = model.get_params()
    summary = {key: params[key] for key in display_keys if key in params}
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        summary["n_iter_"] = int(np.atleast_1d(n_iter).max())
    return summary


def train_model(
    model_type: type[BaseEstimator],
    params: dict[str, Any],
    features,
    target,
) -> BaseEstimator:
    model = model_type(**params)
    model.fit(features, target)
    return model


def evaluate_model(
    model: BaseEstimator,
    features,
    target,
    metrics_fn: MetricsFunction,
) -> Metrics:
    predictions = model.predict(features)
    return metrics_fn(target, predictions)


def apply_preprocessing(spec: ModelSpec, features_train, features_test):
    if spec.preprocessing != "StandardScaler":
        return features_train, features_test

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    return features_train, features_test


def run_pipeline(
    method_name: str,
    dataset: Dataset,
    split: float,
    spec: ModelSpec,
    metrics_fn: MetricsFunction,
    *,
    load_time: float = 0.0,
    stratify: bool = False,
    random_state: int = 42,
) -> MethodResult:
    features_train, features_test, target_train, target_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=split / 100,
        random_state=random_state,
        stratify=dataset.target if stratify else None,
    )

    features_train, features_test = apply_preprocessing(
        spec,
        features_train,
        features_test,
    )

    start = time.perf_counter()
    model = train_model(
        spec.model_type,
        spec.params,
        features_train,
        target_train,
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    metrics = evaluate_model(model, features_test, target_test, metrics_fn)
    test_time = time.perf_counter() - start

    return MethodResult(
        method_name,
        metrics,
        dataset_name=dataset.name,
        n_samples=len(dataset.target),
        n_features=len(dataset.features[0]) if len(dataset.features) else 0,
        n_classes=len(dataset.target_names) if dataset.target_names else None,
        n_train=len(features_train),
        n_test=len(features_test),
        test_split=split,
        load_time=load_time,
        train_time=train_time,
        test_time=test_time,
        random_state=random_state,
        model_params=get_model_params(model, spec.display_keys),
        preprocessing=spec.preprocessing,
        class_names=dataset.target_names or [],
    )
