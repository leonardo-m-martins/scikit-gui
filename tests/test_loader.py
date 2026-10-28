import numpy as np
import pytest

from src.enums import Modes
from src.loader import (
    extract_arff_target_names,
    load_classification_dataset,
    load_dataset,
    load_regression_dataset,
    normalize_target_value,
    transform_string_targets_into_int,
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
def test_normalize_target_value(value, expected):
    assert normalize_target_value(value) == expected


def test_modes_have_gui_values():
    assert Modes.REGRESSION.value == "Regressão"
    assert Modes.CLASSIFICATION.value == "Classificação"


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


def test_load_regression_dataset(arff_factory):
    rows = [
        (0.0, 0.0, 0.5),
        (1.0, 1.0, 1.5),
        (2.0, 2.0, 2.5),
    ]
    path = arff_factory(rows, numeric_class=True, name="regression")

    dataset = load_regression_dataset(str(path))

    assert dataset.name == path.name
    assert isinstance(dataset.features, np.ndarray)
    assert dataset.features.shape == (3, 2)
    assert dataset.features.dtype == float
    np.testing.assert_allclose(dataset.target, [0.5, 1.5, 2.5])
    assert dataset.target_names is None


def test_load_regression_dataset_rejects_non_numeric_target(tmp_path):
    path = tmp_path / "regression.csv"
    path.write_text("0.0,1.0,foo\n1.0,2.0,bar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target must be numeric"):
        load_regression_dataset(str(path))


def test_load_dataset_dispatches_by_mode(tiny_arff, arff_factory):
    regression_path = arff_factory(
        [(0.0, 0.0, 0.5), (1.0, 1.0, 1.5), (2.0, 2.0, 2.5)],
        numeric_class=True,
        name="regression",
    )

    classification = load_dataset(str(tiny_arff), Modes.CLASSIFICATION)
    regression = load_dataset(str(regression_path), Modes.REGRESSION)

    assert classification.target_names == ["yes", "no"]
    assert classification.target.dtype.kind in "iu"
    assert regression.target_names is None
    assert regression.target.dtype == float


def test_load_dataset_rejects_invalid_mode(tiny_arff):
    with pytest.raises(ValueError):
        load_dataset(str(tiny_arff), "invalid")
