import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.ml_clas_sk import (
    format_class_value,
    get_model_params,
    ler_arquivo,
    metricas,
    preparar_classes,
    run_individual_method,
)
from src.shared import ler_arquivo_generico


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"yes", "yes"),
        (1.0, "1"),
        (0.5, "0.5"),
        (2, "2"),
        ("no", "no"),
        (np.float64(1.0), "1"),
    ],
)
def test_format_class_value(value, expected):
    assert format_class_value(value) == expected


def test_preparar_classes_uses_arff_label_order(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)

    classes, names = preparar_classes(["no", "yes", "no"], metadata)

    assert names == ["yes", "no"]
    assert classes.tolist() == [1, 0, 1]


def test_preparar_classes_only_keeps_present_labels(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)

    classes, names = preparar_classes(["yes", "yes"], metadata)

    assert names == ["yes"]
    assert classes.tolist() == [0, 0]


def test_preparar_classes_numeric_falls_back_to_sorted_values(arff_factory):
    rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
        (3.0, 3.0, 0.0),
    ]
    path = arff_factory(rows, numeric_class=True, name="numeric_class")
    _, metadata = ler_arquivo_generico(str(path), com_meta=True)

    classes, names = preparar_classes([1.0, 0.0, 1.0], metadata)

    assert names == ["0", "1"]
    assert classes.tolist() == [1, 0, 1]


def test_preparar_classes_falls_back_when_labels_do_not_match(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)

    classes, names = preparar_classes(["maybe", "yes"], metadata)

    assert names == ["maybe", "yes"]
    assert classes.tolist() == [0, 1]


def test_ler_arquivo(tiny_arff):
    features, classes, class_names = ler_arquivo(str(tiny_arff))

    assert len(features) == len(classes) == 40
    assert all(len(row) == 2 for row in features)
    assert class_names == ["yes", "no"]


def test_metricas_binary_known_values():
    metrics = metricas(
        "PART",
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(5 / 6)
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.f1 == pytest.approx(11 / 15)
    assert metrics.confusion_matrix.tolist() == [[2, 0], [1, 1]]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            MLPClassifier(
                hidden_layer_sizes=(7,),
                activation="tanh",
                solver="lbfgs",
                alpha=0.1,
                learning_rate="adaptive",
                max_iter=42,
            ),
            {
                "hidden_layer_sizes": (7,),
                "activation": "tanh",
                "solver": "lbfgs",
                "alpha": 0.1,
                "learning_rate": "adaptive",
                "max_iter": 42,
            },
        ),
        (
            RandomForestClassifier(
                n_estimators=7,
                criterion="entropy",
                max_depth=3,
                min_samples_split=4,
                min_samples_leaf=2,
            ),
            {
                "n_estimators": 7,
                "criterion": "entropy",
                "max_depth": 3,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
            },
        ),
        (
            DecisionTreeClassifier(criterion="entropy", max_depth=4),
            {"criterion": "entropy", "max_depth": 4},
        ),
        (
            AdaBoostClassifier(n_estimators=7, learning_rate=0.25),
            {"n_estimators": 7, "learning_rate": 0.25},
        ),
    ],
)
def test_get_model_params_supported_models(model, expected):
    params = get_model_params(model)

    for key, value in expected.items():
        assert params[key] == value


def test_get_model_params_unsupported_model():
    assert get_model_params(LogisticRegression()) == {}


def test_get_model_params_reports_best_n_iter():
    class FakeModel:
        n_iter_ = (2, 5, 3)

        def get_params(self):
            return {"criterion": "gini"}

    assert get_model_params(FakeModel()) == {"n_iter_": 5}


@pytest.mark.parametrize(
    "method",
    [
        Methods.RANDOM_FOREST.value,
        Methods.PART.value,
        Methods.ADABOOSTM1.value,
        pytest.param(
            Methods.MULTILAYER_PERCEPTRON.value,
            marks=pytest.mark.filterwarnings(
                "ignore::sklearn.exceptions.ConvergenceWarning"
            ),
        ),
    ],
)
def test_run_individual_method_appends_consistent_result(tiny_arff, method):
    results = []

    run_individual_method(str(tiny_arff), 20, method, results)

    assert len(results) == 1
    result = results[0]
    assert result.method_name == method
    assert result.dataset_name == tiny_arff.name
    assert result.n_samples == 40
    assert result.n_features == 2
    assert result.n_classes == 2
    assert result.class_names == ["yes", "no"]
    assert result.test_split == 20
    assert result.random_state == 42
    assert result.n_train + result.n_test == result.n_samples
    assert 0 < result.n_test < result.n_samples
    assert result.load_time >= 0
    assert result.train_time >= 0
    assert result.test_time >= 0
    assert result.total_time == pytest.approx(
        result.load_time + result.train_time + result.test_time
    )
    assert result.metrics.confusion_matrix.shape == (2, 2)
    assert int(result.metrics.confusion_matrix.sum()) == result.n_test
    assert result.model_params
    if method == Methods.MULTILAYER_PERCEPTRON.value:
        assert result.preprocessing == "StandardScaler"
    else:
        assert result.preprocessing is None


def test_run_individual_method_unsupported_model_is_noop(tiny_arff):
    results = []

    run_individual_method(str(tiny_arff), 20, Methods.SMO.value, results)

    assert results == []


@pytest.mark.parametrize(
    "method",
    [
        Methods.RANDOM_FOREST.value,
        Methods.PART.value,
        Methods.ADABOOSTM1.value,
    ],
)
def test_supported_non_mlp_methods_are_reproducible(tiny_arff, method):
    first = []
    second = []

    run_individual_method(str(tiny_arff), 20, method, first)
    run_individual_method(str(tiny_arff), 20, method, second)

    assert first[0].metrics.accuracy == second[0].metrics.accuracy
    assert first[0].metrics.confusion_matrix.tolist() == (
        second[0].metrics.confusion_matrix.tolist()
    )
