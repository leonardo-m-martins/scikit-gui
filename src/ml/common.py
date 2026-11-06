import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.enums import Methods, Preprocessors
from src.shared import MethodResult, Metrics


@dataclass(frozen=True)
class ModelSpec:
    model_type: type[BaseEstimator]
    params: dict[str, Any] = field(default_factory=dict)
    preprocessing: Preprocessors = Preprocessors.NONE

    def update_params(
        self,
        params: dict[str, Any],
        *,
        preprocessing: Preprocessors | None = None,
    ) -> "ModelSpec":
        return replace(
            self,
            params={**self.params, **params},
            preprocessing=(
                self.preprocessing if preprocessing is None else preprocessing
            ),
        )


PREPROCESSOR_TYPES = {
    Preprocessors.NONE: None,
    Preprocessors.STANDARD_SCALER: StandardScaler,
    Preprocessors.MIN_MAX_SCALER: MinMaxScaler,
    Preprocessors.ROBUST_SCALER: RobustScaler,
}


@dataclass(frozen=True)
class Dataset:
    name: str
    features: Any
    target: np.ndarray
    load_time: float
    target_names: list[str] | None = None
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""


MetricsFunction = Callable[[np.ndarray, np.ndarray], Metrics]


@dataclass(frozen=True)
class ModeConfig:
    specs: dict[Methods, ModelSpec]
    metrics_fn: MetricsFunction


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
    preprocessor_type = PREPROCESSOR_TYPES[spec.preprocessing]

    if preprocessor_type is None:
        return features_train, features_test

    preprocessor = preprocessor_type()
    features_train = preprocessor.fit_transform(features_train)
    features_test = preprocessor.transform(features_test)
    return features_train, features_test


def run_method(
    method: Methods,
    dataset: Dataset,
    split: float,
    mode_config: ModeConfig,
    *,
    stratify: bool = False,
    random_state: int = 42,
) -> MethodResult:
    spec = mode_config.specs.get(method)
    if spec is None:
        print(f"Method {method} is not supported!")
        return
    metrics_fn = partial(mode_config.metrics_fn, labels=dataset.target_names)

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
        method,
        metrics,
        dataset_name=dataset.name,
        n_samples=len(dataset.target),
        n_features=len(dataset.features[0]) if len(dataset.features) else 0,
        n_classes=len(dataset.target_names) if dataset.target_names else None,
        n_train=len(features_train),
        n_test=len(features_test),
        test_split=split,
        load_time=dataset.load_time,
        train_time=train_time,
        test_time=test_time,
        random_state=random_state,
        model_params=model.get_params(),
        preprocessing=(
            spec.preprocessing.value
            if spec.preprocessing is not Preprocessors.NONE
            else None
        ),
        class_names=dataset.target_names or [],
    )


def run_all_methods(
    dataset: Dataset,
    split: float,
    mode_config: ModeConfig,
    *,
    stratify: bool = False,
    random_state: int = 42,
    on_method_start: Callable[[str], None] | None = None,
    on_method_complete: Callable[[], None] | None = None,
) -> list[MethodResult]:
    results = []
    for method in mode_config.specs:
        if on_method_start is not None:
            on_method_start(method.value)

        result = run_method(
            method.value,
            dataset,
            split,
            mode_config,
            stratify=stratify,
            random_state=random_state,
        )
        if result is not None:
            results.append(result)

        if on_method_complete is not None:
            on_method_complete()

    return results
