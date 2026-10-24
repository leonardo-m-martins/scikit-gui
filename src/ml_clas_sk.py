import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler
from src.enums import Methods
from src.shared import (
    MethodResult,
    Metrics,
    ClassificationMetrics,
    ler_arquivo_generico,
)
from typing import List


def metricas(texto, y_teste, y_pred) -> ClassificationMetrics:
    cm = me.confusion_matrix(y_teste, y_pred)
    ac = me.accuracy_score(y_teste, y_pred)
    pr = me.precision_score(y_teste, y_pred, average="macro")
    re = me.recall_score(y_teste, y_pred, average="macro")
    f1 = me.f1_score(y_teste, y_pred, average="macro")
    return ClassificationMetrics(ac, pr, re, f1, cm)


def ler_arquivo(arquivo):
    df = ler_arquivo_generico(arquivo)

    atributos = df.iloc[:, :-1].astype(float).values.tolist()
    classes = df.iloc[:, -1].tolist()

    return atributos, classes


def mlp_treino(x_train, y_train):
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(125,),
        activation="logistic",
        solver="adam",
        alpha=0.01,
        learning_rate="adaptive",
        max_iter=10000,
    )
    mlp_model.fit(x_train, y_train)

    return mlp_model


def rf_treino(x_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(x_train, y_train)
    return rf_model


def decision_tree_training(feat_train, class_train):
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(feat_train, class_train)
    return model


def ada_boost_training(feat_train, class_train):
    model = AdaBoostClassifier(random_state=42)
    model.fit(feat_train, class_train)
    return model


SUPPORTED_MODELS = (
    MLPClassifier | RandomForestClassifier | DecisionTreeClassifier | AdaBoostClassifier
)

MODEL_PARAM_KEYS = {
    MLPClassifier: (
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "learning_rate",
        "max_iter",
    ),
    RandomForestClassifier: (
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    ),
    DecisionTreeClassifier: ("criterion", "max_depth"),
    AdaBoostClassifier: ("n_estimators", "learning_rate"),
}


def get_model_params(model: SUPPORTED_MODELS) -> dict:
    keys = MODEL_PARAM_KEYS.get(type(model), ())
    params = model.get_params()
    summary = {key: params[key] for key in keys if key in params}
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        summary["n_iter_"] = int(np.atleast_1d(n_iter).max())
    return summary


def test_model(
    features, classes, model: SUPPORTED_MODELS, name: str
) -> ClassificationMetrics:
    pred = model.predict(features)
    return metricas(name, classes, pred)


def run_individual_method(
    filename: str, split, method: Methods, METHOD_RESULTS: List[MethodResult]
):
    start = time.perf_counter()
    features, classes_str = ler_arquivo(filename)
    load_time = time.perf_counter() - start

    classes = list(set(classes_str))
    classes = np.array([classes.index(c) for c in classes_str])
    features_training, features_test, classes_training, classes_test = train_test_split(
        features, classes, test_size=split / 100, random_state=42, stratify=classes
    )

    match method:
        case Methods.RANDOM_FOREST:
            train_model = rf_treino
        case Methods.MULTILAYER_PERCEPTRON:
            scaler = StandardScaler()
            features_training = scaler.fit_transform(features_training)
            features_test = scaler.fit_transform(features_test)
            train_model = mlp_treino
        case Methods.PART:
            train_model = decision_tree_training
        case Methods.ADABOOSTM1:
            train_model = ada_boost_training
        case _:
            print(f"Método {method} não suportado!")
            return

    start = time.perf_counter()
    model = train_model(features_training, classes_training)
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
        n_classes=len(set(classes_str)),
        n_train=len(features_training),
        n_test=len(features_test),
        test_split=split,
        load_time=load_time,
        train_time=train_time,
        test_time=test_time,
        random_state=42,
        model_params=get_model_params(model),
        preprocessing=(
            "StandardScaler" if method == Methods.MULTILAYER_PERCEPTRON else None
        ),
    )
    METHOD_RESULTS.append(results)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: List[MethodResult]):
    for method in Methods:
        run_individual_method(filename, split, method, METHOD_RESULTS)
