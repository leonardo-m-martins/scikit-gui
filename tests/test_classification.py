import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.enums import Methods
from src.ml_clas_sk import (
    MODEL_SPECS,
    classification_metrics,
    extract_arff_target_names,
    load_classification_dataset,
    normalize_target_value,
    run_individual_method,
    transform_string_targets_into_int,
)
from src.ml_common import get_model_params
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
def test_normalize_target_value(value, expected):
    assert normalize_target_value(value) == expected


def test_extract_arff_target_names(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)

    assert extract_arff_target_names(metadata) == ["yes", "no"]
    assert extract_arff_target_names(None) is None


def test_transform_string_targets_into_int_uses_arff_order(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)
    names = extract_arff_target_names(metadata)

    classes, class_names = transform_string_targets_into_int(
        ["no", "yes", "no"],
        names,
    )

    assert class_names == ["yes", "no"]
    assert classes.tolist() == [1, 0, 1]


def test_transform_string_targets_into_int_only_keeps_present_labels(tiny_arff):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)
    names = extract_arff_target_names(metadata)

    classes, class_names = transform_string_targets_into_int(
        ["yes", "yes"],
        names,
    )

    assert class_names == ["yes"]
    assert classes.tolist() == [0, 0]


def test_transform_string_targets_into_int_falls_back_to_sorted_values(
    arff_factory,
):
    rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 1.0),
        (3.0, 3.0, 0.0),
    ]
    path = arff_factory(rows, numeric_class=True, name="numeric_class")
    _, metadata = ler_arquivo_generico(str(path), com_meta=True)

    classes, class_names = transform_string_targets_into_int(
        [1.0, 0.0, 1.0],
        extract_arff_target_names(metadata),
    )

    assert class_names == ["0", "1"]
    assert classes.tolist() == [1, 0, 1]


def test_transform_string_targets_into_int_falls_back_when_labels_do_not_match(
    tiny_arff,
):
    _, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)
    names = extract_arff_target_names(metadata)

    classes, class_names = transform_string_targets_into_int(
        ["maybe", "yes"],
        names,
    )

    assert class_names == ["maybe", "yes"]
    assert classes.tolist() == [0, 1]


def test_load_classification_dataset(tiny_arff):
    dataset = load_classification_dataset(str(tiny_arff))

    assert dataset.name == tiny_arff.name
    assert isinstance(dataset.features, np.ndarray)
    assert dataset.features.shape == (40, 2)
    assert dataset.features.dtype == float
    assert dataset.target.shape == (40,)
    assert dataset.target.dtype.kind in "iu"
    assert dataset.target_names == ["yes", "no"]


def test_load_classification_dataset_rejects_single_class(arff_factory):
    path = arff_factory([(0.0, 0.0, "yes"), (1.0, 1.0, "yes")])

    with pytest.raises(ValueError, match="at least two classes"):
        load_classification_dataset(str(path))


def test_load_classification_dataset_rejects_non_numeric_features(tmp_path):
    path = tmp_path / "categorical.csv"
    path.write_text("feature,class\nfoo,yes\nbar,no\n", encoding="utf-8")

    with pytest.raises(ValueError, match="features must be numeric"):
        load_classification_dataset(str(path))


def test_classification_metrics_binary_known_values():
    metrics = classification_metrics(
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
    spec = next(spec for spec in MODEL_SPECS.values() if spec.model_type is type(model))
    params = get_model_params(model, spec.display_keys)

    for key, value in expected.items():
        assert params[key] == value


@pytest.mark.parametrize(
    "method",
    [
        Methods.RANDOM_FOREST.value,
        Methods.DECISION_TREE.value,
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
        Methods.DECISION_TREE.value,
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
