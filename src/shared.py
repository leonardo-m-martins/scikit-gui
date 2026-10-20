from dataclasses import dataclass
from typing import List, Any
from datetime import datetime


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
