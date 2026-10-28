import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.ml_common import (
    Dataset,
    ModelSpec,
    apply_preprocessing,
    evaluate_model,
    get_model_params,
    run_pipeline,
    train_model,
)
from src.shared import ClassificationMetrics, RegressionMetrics


def test_get_model_params_uses_display_keys():
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    params = get_model_params(model, ("criterion", "max_depth"))

    assert params == {"criterion": "entropy", "max_depth": 3}


def test_get_model_params_without_display_keys():
    assert get_model_params(DecisionTreeClassifier()) == {}


def test_get_model_params_reports_best_n_iter():
    class FakeModel:
        n_iter_ = (2, 5, 3)

        def get_params(self):
            return {"criterion": "gini"}

    assert get_model_params(FakeModel()) == {"n_iter_": 5}


def test_train_model_instantiates_and_fits():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([0, 0, 1, 1])

    model = train_model(
        DecisionTreeClassifier,
        {"criterion": "entropy", "max_depth": 2},
        features,
        target,
    )

    assert isinstance(model, DecisionTreeClassifier)
    assert model.criterion == "entropy"
    assert model.max_depth == 2
    assert model.predict(features).tolist() == target.tolist()


def test_evaluate_model_uses_metrics_function():
    class FakeModel:
        def predict(self, features):
            assert features == [[1.0], [2.0]]
            return np.array([1, 0])

    received = []

    def metrics_fn(target, predictions):
        received.append((target, predictions))
        return ClassificationMetrics(1.0, 1.0, 1.0, 1.0, np.zeros((2, 2)))

    metrics = evaluate_model(
        FakeModel(),
        [[1.0], [2.0]],
        np.array([1, 1]),
        metrics_fn,
    )

    assert isinstance(metrics, ClassificationMetrics)
    assert len(received) == 1
    assert received[0][1].tolist() == [1, 0]


def test_apply_preprocessing_fits_only_on_train():
    features_train = np.array([[0.0], [2.0]])
    features_test = np.array([[10.0], [20.0]])
    spec = ModelSpec(DecisionTreeClassifier, preprocessing="StandardScaler")

    transformed_train, transformed_test = apply_preprocessing(
        spec,
        features_train,
        features_test,
    )

    scaler = StandardScaler()
    expected_train = scaler.fit_transform(features_train)
    expected_test = scaler.transform(features_test)

    np.testing.assert_allclose(transformed_train, expected_train)
    np.testing.assert_allclose(transformed_test, expected_test)


def test_run_pipeline_builds_complete_result():
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dataset = Dataset("tiny.arff", features, target, ["no", "yes"])
    spec = ModelSpec(
        DecisionTreeClassifier,
        {"criterion": "entropy", "max_depth": 1},
        ("criterion", "max_depth"),
    )

    def metrics_fn(y_true, y_pred):
        return ClassificationMetrics(1.0, 1.0, 1.0, 1.0, np.zeros((2, 2)))

    result = run_pipeline(
        "Decision Tree",
        dataset,
        25,
        spec,
        metrics_fn,
        load_time=0.1,
        stratify=True,
    )

    assert result.method_name == "Decision Tree"
    assert result.dataset_name == "tiny.arff"
    assert result.n_samples == 8
    assert result.n_features == 2
    assert result.n_classes == 2
    assert result.n_train + result.n_test == 8
    assert result.test_split == 25
    assert result.random_state == 42
    assert result.class_names == ["no", "yes"]
    assert result.load_time == pytest.approx(0.1)
    assert result.model_params == {"criterion": "entropy", "max_depth": 1}
    assert isinstance(result.metrics, ClassificationMetrics)


def test_run_pipeline_omits_class_metadata_without_target_names():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([0.0, 1.0, 2.0, 3.0])
    dataset = Dataset("regression.arff", features, target)
    spec = ModelSpec(DecisionTreeRegressor)

    def metrics_fn(y_true, y_pred):
        return RegressionMetrics(0.0, 0.0, 0.0, 1.0)

    result = run_pipeline(
        "Tree",
        dataset,
        25,
        spec,
        metrics_fn,
    )

    assert result.n_classes is None
    assert result.class_names == []
    assert isinstance(result.metrics, RegressionMetrics)
