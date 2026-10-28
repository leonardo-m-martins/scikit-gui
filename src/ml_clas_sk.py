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


def metricas(y_teste, y_pred) -> ClassificationMetrics:
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


def run_individual_method(
    filename: str, split, method: Methods, METHOD_RESULTS: list[MethodResult]
):
    spec = MODEL_SPECS.get(method)
    if spec is None:
        print(f"Método {method} não suportado!")
        return

    start = time.perf_counter()
    features, classes, class_names = ler_arquivo(filename)
    load_time = time.perf_counter() - start

    dataset = Dataset(
        name=Path(filename).name,
        features=features,
        target=classes,
        target_names=class_names,
    )

    result = run_pipeline(
        method,
        dataset,
        split,
        spec,
        metricas,
        load_time=load_time,
        stratify=True,
    )
    METHOD_RESULTS.append(result)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: list[MethodResult]):
    for method in Methods.values():
        run_individual_method(filename, split, method, METHOD_RESULTS)
