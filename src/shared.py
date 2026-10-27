from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from scipy.io import arff


class Metrics:
    pass


@dataclass
class ClassificationMetrics(Metrics):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Any


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0.1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} min {remainder:.2f} s"


@dataclass(kw_only=True, eq=False)
class MethodResult:
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

    @property
    def total_time(self) -> float | None:
        if None in (self.load_time, self.train_time, self.test_time):
            return None
        return self.load_time + self.train_time + self.test_time

    def get_display_name(self):
        return f"{self.timestamp.strftime('%H:%M:%S')} - {self.method_name}"


def ler_arquivo_generico(arquivo: str, com_meta: bool = False):
    if arquivo.endswith(".arff"):
        data, meta = arff.loadarff(arquivo)
        df = pd.DataFrame(data)
        # Decodificar bytes para string nas colunas object/bytes
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        return (df, meta) if com_meta else df
    else:
        df = pd.read_csv(arquivo, header=None)
        return (df, None) if com_meta else df
