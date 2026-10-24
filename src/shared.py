from dataclasses import dataclass
from typing import List, Any
from datetime import datetime
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


class MethodResult:
    def __init__(
        self,
        method_name: str,
        metrics: Metrics,
        *,
        dataset_name: str = "-",
        n_samples: int | None = None,
        n_features: int | None = None,
        n_classes: int | None = None,
        n_train: int | None = None,
        n_test: int | None = None,
        test_split: float | None = None,
        load_time: float | None = None,
        train_time: float | None = None,
        test_time: float | None = None,
        random_state: int | None = None,
        model_params: dict | None = None,
        preprocessing: str | None = None,
    ):
        self.method_name = method_name
        self.metrics = metrics
        self.timestamp = datetime.now()
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_train = n_train
        self.n_test = n_test
        self.test_split = test_split
        self.load_time = load_time
        self.train_time = train_time
        self.test_time = test_time
        self.random_state = random_state
        self.model_params = model_params or {}
        self.preprocessing = preprocessing

    @property
    def total_time(self) -> float | None:
        if None in (self.load_time, self.train_time, self.test_time):
            return None
        return self.load_time + self.train_time + self.test_time

    def get_display_name(self):
        return f"{self.timestamp.strftime('%H:%M:%S')} - {self.method_name}"


def ler_arquivo_generico(arquivo: str):
    if arquivo.endswith(".arff"):
        data, meta = arff.loadarff(arquivo)
        df = pd.DataFrame(data)
        # Decodificar bytes para string nas colunas object/bytes
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        return df
    else:
        return pd.read_csv(arquivo, header=None)
