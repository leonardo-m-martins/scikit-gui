import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics as me
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.shared import ClassificationMetrics, MethodResult, ler_arquivo_generico


@dataclass(frozen=True)
class ModelSpec:
    model_type: type[BaseEstimator]
    params: dict[str, Any] = field(default_factory=dict)
    display_keys: tuple[str, ...] = ()
    preprocessing: str | None = None


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

MODEL_PARAM_KEYS = {spec.model_type: spec.display_keys for spec in MODEL_SPECS.values()}


def metricas(texto, y_teste, y_pred) -> ClassificationMetrics:
    cm = me.confusion_matrix(y_teste, y_pred)
    ac = me.accuracy_score(y_teste, y_pred)
    pr = me.precision_score(y_teste, y_pred, average="macro")
    re = me.recall_score(y_teste, y_pred, average="macro")
    f1 = me.f1_score(y_teste, y_pred, average="macro")
    return ClassificationMetrics(ac, pr, re, f1, cm)


def format_class_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def preparar_classes(values, meta):
    present = set(values)

    if meta is not None:
        attribute = meta.names()[-1]
        kind, labels = meta[attribute]
        if kind == "nominal" and labels:
            labels = [format_class_value(label) for label in labels]
            if present <= set(labels):
                selected = [label for label in labels if label in present]
                index = {label: i for i, label in enumerate(selected)}
                return np.array([index[value] for value in values]), selected

    ordered = sorted(present)
    names = [format_class_value(value) for value in ordered]
    index = {value: i for i, value in enumerate(ordered)}
    return np.array([index[value] for value in values]), names


def ler_arquivo(arquivo):
    df, meta = ler_arquivo_generico(arquivo, com_meta=True)

    atributos = df.iloc[:, :-1].astype(float).values.tolist()
    classes, class_names = preparar_classes(df.iloc[:, -1].tolist(), meta)

    return atributos, classes, class_names


def get_model_params(model: BaseEstimator) -> dict:
    keys = MODEL_PARAM_KEYS.get(type(model), ())
    params = model.get_params()
    summary = {key: params[key] for key in keys if key in params}
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        summary["n_iter_"] = int(np.atleast_1d(n_iter).max())
    return summary


def train_model(
    model_type: type[BaseEstimator],
    params: dict[str, Any],
    features,
    classes,
) -> BaseEstimator:
    model = model_type(**params)
    model.fit(features, classes)
    return model


def test_model(
    features, classes, model: BaseEstimator, name: str
) -> ClassificationMetrics:
    pred = model.predict(features)
    return metricas(name, classes, pred)


def run_individual_method(
    filename: str, split, method: Methods, METHOD_RESULTS: list[MethodResult]
):
    start = time.perf_counter()
    features, classes, class_names = ler_arquivo(filename)
    load_time = time.perf_counter() - start

    features_training, features_test, classes_training, classes_test = train_test_split(
        features, classes, test_size=split / 100, random_state=42, stratify=classes
    )

    spec = MODEL_SPECS.get(method)
    if spec is None:
        print(f"Método {method} não suportado!")
        return

    if spec.preprocessing == "StandardScaler":
        scaler = StandardScaler()
        features_training = scaler.fit_transform(features_training)
        features_test = scaler.transform(features_test)

    start = time.perf_counter()
    model = train_model(
        spec.model_type, spec.params, features_training, classes_training
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    metrics = test_model(features_test, classes_test, model, method)
    test_time = time.perf_counter() - start

    results = MethodResult(
        method,
        metrics,
        dataset_name=Path(filename).name,
        n_samples=len(classes),
        n_features=len(features[0]) if features else 0,
        n_classes=len(class_names),
        n_train=len(features_training),
        n_test=len(features_test),
        test_split=split,
        load_time=load_time,
        train_time=train_time,
        test_time=test_time,
        random_state=42,
        model_params=get_model_params(model),
        preprocessing=spec.preprocessing,
        class_names=class_names,
    )
    METHOD_RESULTS.append(results)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: list[MethodResult]):
    for method in Methods.values():
        run_individual_method(filename, split, method, METHOD_RESULTS)
