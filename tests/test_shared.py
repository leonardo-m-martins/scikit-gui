from datetime import datetime

import pytest

from src.enums import MethodResultField
from src.shared import (
    ClassificationMetrics,
    MethodResult,
    format_metadata,
    ler_arquivo_generico,
)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (MethodResultField.method_name, "PART", "PART"),
        (MethodResultField.dataset_name, None, "-"),
        (MethodResultField.n_samples, 40, "40"),
        (MethodResultField.n_samples, None, "-"),
        (MethodResultField.test_split, 50, "50%"),
        (MethodResultField.test_split, None, "-"),
        (MethodResultField.train_test, (20, 20), "20 / 20"),
        (MethodResultField.train_test, (None, None), "-"),
        (MethodResultField.load_time, None, "-"),
        (MethodResultField.load_time, 0.0, "0.00 ms"),
        (MethodResultField.load_time, 0.099, "99.00 ms"),
        (MethodResultField.load_time, 0.1, "0.10 s"),
        (MethodResultField.load_time, 59.99, "59.99 s"),
        (MethodResultField.load_time, 60.0, "1 min 0.00 s"),
        (MethodResultField.load_time, 72.4, "1 min 12.40 s"),
        (
            MethodResultField.timestamp,
            datetime(2026, 10, 24, 15, 3, 9),
            "24/10/2026 15:03:09",
        ),
        (MethodResultField.timestamp, None, "-"),
        (MethodResultField.model_params, {"criterion": "entropy"}, "criterion=entropy"),
        (MethodResultField.model_params, {}, "-"),
    ],
)
def test_format_metadata(field, value, expected):
    assert format_metadata(field, value) == expected


def test_method_result_defaults():
    metrics = ClassificationMetrics(1.0, 1.0, 1.0, 1.0, None)
    result = MethodResult("PART", metrics)

    assert result.dataset_name == "-"
    assert result.n_samples is None
    assert result.n_features is None
    assert result.n_classes is None
    assert result.n_train is None
    assert result.n_test is None
    assert result.test_split is None
    assert result.load_time is None
    assert result.train_time is None
    assert result.test_time is None
    assert result.random_state is None
    assert result.model_params == {}
    assert result.preprocessing is None
    assert result.class_names == []
    assert result.total_time is None


def test_method_result_fields_match_enum(result_factory):
    result = result_factory()

    for field in MethodResultField:
        assert hasattr(result, field.name)


def test_method_result_indexes_fields(result_factory):
    result = result_factory()

    assert result[MethodResultField.method_name] == "PART"
    assert result[MethodResultField.train_test] == (20, 20)


def test_method_result_total_time():
    result = MethodResult("PART", ClassificationMetrics(1.0, 1.0, 1.0, 1.0, None))
    result.load_time = 0.1
    result.train_time = 0.2
    result.test_time = 0.3

    assert result.total_time == pytest.approx(0.6)

    result.test_time = None
    assert result.total_time is None


def test_method_result_display_name():
    result = MethodResult("PART", ClassificationMetrics(1.0, 1.0, 1.0, 1.0, None))
    result.timestamp = datetime(2026, 10, 24, 15, 3, 9)

    assert result.get_display_name() == "15:03:09 - PART"


def test_ler_arquivo_generico_arff(tiny_arff):
    dataframe = ler_arquivo_generico(str(tiny_arff))

    assert list(dataframe.columns) == ["f1", "f2", "class"]
    assert set(dataframe["class"]) == {"yes", "no"}
    assert all(isinstance(value, str) for value in dataframe["class"])


def test_ler_arquivo_generico_arff_with_metadata(tiny_arff):
    dataframe, metadata = ler_arquivo_generico(str(tiny_arff), com_meta=True)

    assert len(dataframe) == 40
    assert metadata.names()[-1] == "class"
    kind, labels = metadata["class"]
    assert kind == "nominal"
    assert list(labels) == ["yes", "no"]
