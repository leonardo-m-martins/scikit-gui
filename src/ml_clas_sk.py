import time
from pathlib import Path

import numpy as np
from sklearn import metrics as me
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.ml_common import Dataset, ModelSpec, run_pipeline
from src.shared import ClassificationMetrics, MethodResult, ler_arquivo_generico

MODEL_SPECS: dict[Methods, ModelSpec] = {
    Methods.MULTILAYER_PERCEPTRON: ModelSpec(
        MLPClassifier,
        {
            "hidden_layer_sizes": (125,),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 0.01,
            "learning_rate": "adaptive",
            "max_iter": 10000,
        },
        (
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate",
            "max_iter",
        ),
        preprocessing="StandardScaler",
    ),
    Methods.RANDOM_FOREST: ModelSpec(
        RandomForestClassifier,
        {
            "n_estimators": 100,
            "criterion": "gini",
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
        ),
    ),
    Methods.DECISION_TREE: ModelSpec(
        DecisionTreeClassifier,
        {"criterion": "entropy", "random_state": 42},
        ("criterion", "max_depth"),
    ),
    Methods.ADABOOSTM1: ModelSpec(
        AdaBoostClassifier,
        {"random_state": 42},
        ("n_estimators", "learning_rate"),
    ),
    Methods.GAUSSIAN_NAIVE_BAYES: ModelSpec(GaussianNB),
    Methods.MULTI_NAIVE_BAYES: ModelSpec(MultinomialNB),
    Methods.COMPLEMENT_NAIVE_BAYES: ModelSpec(ComplementNB),
    Methods.LOGISTICS: ModelSpec(
        LogisticRegression,
        {"random_state": 42, "max_iter": 10000},
        ("max_iter"),
        preprocessing="StandardScaler",
    ),
}


def classification_metrics(y_true, y_pred) -> ClassificationMetrics:
    cm = me.confusion_matrix(y_true, y_pred)
    ac = me.accuracy_score(y_true, y_pred)
    pr = me.precision_score(y_true, y_pred, average="macro")
    re = me.recall_score(y_true, y_pred, average="macro")
    f1 = me.f1_score(y_true, y_pred, average="macro")
    return ClassificationMetrics(ac, pr, re, f1, cm)


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


def load_classification_dataset(filename: str) -> Dataset:
    frame, meta = ler_arquivo_generico(filename, com_meta=True)

    target_column = frame.columns[-1]
    target_values = frame[target_column]
    _validate_classification_target(target_values)

    feature_frame = frame.drop(columns=target_column)
    try:
        features = feature_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Classification features must be numeric or binary numeric columns."
        ) from exc

    if features.size == 0:
        raise ValueError("Classification features must not be empty.")

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


def run_individual_method(
    filename: str, split, method: Methods, METHOD_RESULTS: list[MethodResult]
):
    spec = MODEL_SPECS.get(method)
    if spec is None:
        print(f"Method {method} is not supported!")
        return

    start = time.perf_counter()
    dataset = load_classification_dataset(filename)
    load_time = time.perf_counter() - start

    result = run_pipeline(
        method,
        dataset,
        split,
        spec,
        classification_metrics,
        load_time=load_time,
        stratify=True,
    )
    METHOD_RESULTS.append(result)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: list[MethodResult]):
    for method in Methods.values():
        run_individual_method(filename, split, method, METHOD_RESULTS)
