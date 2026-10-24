import time
from pathlib import Path

import numpy as np
from sklearn import metrics as me
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.shared import ClassificationMetrics, MethodResult, ler_arquivo_generico


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
    filename: str, split, method: Methods, METHOD_RESULTS: list[MethodResult]
):
    start = time.perf_counter()
    features, classes, class_names = ler_arquivo(filename)
    load_time = time.perf_counter() - start

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
        n_classes=len(class_names),
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
        class_names=class_names,
    )
    METHOD_RESULTS.append(results)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: list[MethodResult]):
    for method in Methods:
        run_individual_method(filename, split, method, METHOD_RESULTS)
