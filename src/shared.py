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


class MethodResult:
    def __init__(self, method_name: str, metrics: Metrics):
        self.method_name = method_name
        self.metrics = metrics
        self.timestamp = datetime.now()

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
