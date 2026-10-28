from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.enums import MethodResultField


class Metrics:
    pass


@dataclass
class ClassificationMetrics(Metrics):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Any


@dataclass
class RegressionMetrics(Metrics):
    mae: float
    mse: float
    rmse: float
    r2: float


class AbstractResult[FieldT: Enum](ABC):
    @classmethod
    @abstractmethod
    def fields(cls) -> type[FieldT]:
        raise NotImplementedError

    def __getitem__(self, field: FieldT) -> object:
        return getattr(self, field.name)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0.1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} min {remainder:.2f} s"


def format_metadata(key: MethodResultField, value: Any) -> str:
    match key:
        case MethodResultField.timestamp:
            return "-" if value is None else value.strftime("%d/%m/%Y %H:%M:%S")

        case MethodResultField.test_split:
            return "-" if value is None else f"{value:g}%"

        case MethodResultField.train_test:
            if not value:
                return "-"
            n_train, n_test = value
            if n_train is None or n_test is None:
                return "-"
            return f"{n_train} / {n_test}"

        case (
            MethodResultField.load_time
            | MethodResultField.train_time
            | MethodResultField.test_time
            | MethodResultField.total_time
        ):
            return _format_duration(value)

        case MethodResultField.model_params:
            return (
                ", ".join(
                    f"{model_key}={model_value}"
                    for model_key, model_value in (value or {}).items()
                )
                or "-"
            )

        case _:
            return "-" if value is None else str(value)


@dataclass(kw_only=True, eq=False)
class MethodResult(AbstractResult[MethodResultField]):
    method_name: str = field(kw_only=False)
    metrics: Metrics = field(kw_only=False)
    dataset_name: str = "-"
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    n_train: int | None = None
    n_test: int | None = None
    test_split: float | None = None
    load_time: float | None = None
    train_time: float | None = None
    test_time: float | None = None
    random_state: int | None = None
    model_params: dict = field(default_factory=dict)
    preprocessing: str | None = None
    class_names: list[str] = field(default_factory=list)
    timestamp: datetime = field(init=False, default_factory=datetime.now)

    @classmethod
    def fields(cls) -> type[MethodResultField]:
        return MethodResultField

    @property
    def total_time(self) -> float | None:
        if None in (self.load_time, self.train_time, self.test_time):
            return None
        return self.load_time + self.train_time + self.test_time

    @property
    def train_test(self) -> tuple[int | None, int | None]:
        return self.n_train, self.n_test

    def get_display_name(self):
        return f"{self.timestamp.strftime('%H:%M:%S')} - {self.method_name}"
