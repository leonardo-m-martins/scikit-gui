from functools import partial

from sklearn import metrics as me
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.ml_common import Dataset, ModelSpec, run_pipeline
from src.shared import ClassificationMetrics, MethodResult

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


def classification_metrics(y_true, y_pred, labels=None) -> ClassificationMetrics:
    if labels is not None:
        present = set(y_true) | set(y_pred)
        labels = [label for label in labels if label in present]

    cm = me.confusion_matrix(y_true, y_pred, labels=labels)
    ac = me.accuracy_score(y_true, y_pred)
    pr = me.precision_score(y_true, y_pred, average="macro", labels=labels)
    re = me.recall_score(y_true, y_pred, average="macro", labels=labels)
    f1 = me.f1_score(y_true, y_pred, average="macro", labels=labels)
    return ClassificationMetrics(ac, pr, re, f1, cm)


def run_individual_method(
    dataset: Dataset, split, method: Methods, METHOD_RESULTS: list[MethodResult]
):
    spec = MODEL_SPECS.get(method)
    if spec is None:
        print(f"Method {method} is not supported!")
        return

    metrics_fn = partial(
        classification_metrics,
        labels=dataset.target_names,
    )

    result = run_pipeline(
        method,
        dataset,
        split,
        spec,
        metrics_fn,
        load_time=dataset.load_time,
        stratify=True,
    )
    METHOD_RESULTS.append(result)


def run_all_methods(filename: str, split: int, METHOD_RESULTS: list[MethodResult]):
    for method in Methods.values():
        run_individual_method(filename, split, method, METHOD_RESULTS)
