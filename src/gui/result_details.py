"""Panels that display details and metrics for a selected result."""

import tkinter as tk
from tkinter import ttk

from src.enums import MethodResultField
from src.gui.styles import (
    STYLE_MATRIX_CELL,
    STYLE_MATRIX_HEADER,
    STYLE_METADATA,
    STYLE_METRIC,
    STYLE_SECTION,
)
from src.shared import (
    ClassificationMetrics,
    MethodResult,
    RegressionMetrics,
    format_metadata,
)

METADATA_FIELDS = tuple(
    field for field in MethodResultField if field is not MethodResultField.model_params
)

METRIC_NAMES = {
    "acc": "Acurácia",
    "prec": "Precisão",
    "rec": "Recall",
    "f1": "F1-measure",
}


class DetailsPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, weight=1)

        self.metadata: dict[MethodResultField, ttk.Label] = {}
        for index, field in enumerate(METADATA_FIELDS):
            row, column = divmod(index, 2)
            ttk.Label(self, text=field.value, style=STYLE_METADATA).grid(
                row=row, column=column * 2, sticky=tk.W, padx=(0, 5), pady=2
            )
            label = ttk.Label(self, text="-")
            label.grid(
                row=row, column=column * 2 + 1, sticky=tk.W, padx=(0, 20), pady=2
            )
            self.metadata[field] = label

        params_row = (len(METADATA_FIELDS) + 1) // 2
        ttk.Label(
            self,
            text=MethodResultField.model_params.value,
            style=STYLE_METADATA,
        ).grid(row=params_row, column=0, sticky=tk.NW, pady=2)
        params_label = ttk.Label(self, text="-", wraplength=520, justify=tk.LEFT)
        params_label.grid(row=params_row, column=1, columnspan=3, sticky=tk.W, pady=2)
        self.metadata[MethodResultField.model_params] = params_label

    def render(self, result: MethodResult):
        for field, label in self.metadata.items():
            label.config(text=format_metadata(field, result[field]))


class ConfusionMatrixView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

    def clear(self):
        for child in self.winfo_children():
            child.destroy()
        ttk.Label(self, text="-").grid(row=0, column=0, sticky=tk.W)

    def render(self, matrix, class_names):
        for child in self.winfo_children():
            child.destroy()

        rows = [list(row) for row in matrix]
        size = len(rows)
        if not size:
            self.clear()
            return

        names = list(class_names or [])
        if len(names) != size:
            names = [f"Classe {index}" for index in range(size)]

        self._make_cell("Real \\ Previsto", 0, 0, bold=True)
        for column, name in enumerate(names, start=1):
            self._make_cell(name, 0, column, bold=True)
        for row, name in enumerate(names, start=1):
            self._make_cell(name, row, 0, bold=True)
            for column in range(size):
                self._make_cell(str(int(rows[row - 1][column])), row, column + 1)

        for column in range(size + 1):
            self.columnconfigure(column, weight=1)

    def _make_cell(self, text, row, column, bold=False):
        cell = ttk.Label(
            self,
            text=text,
            style=STYLE_MATRIX_HEADER if bold else STYLE_MATRIX_CELL,
            relief="solid",
            borderwidth=1,
            padding=(8, 4),
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        cell.grid(row=row, column=column, sticky="nsew")


class MetricsPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text=" 📊 Métricas (média macro) ", padding=10)

        self.metric_labels = {
            key: ttk.Label(self, text=f"{name}: -", style=STYLE_METRIC)
            for key, name in METRIC_NAMES.items()
        }
        for label in self.metric_labels.values():
            label.pack(anchor=tk.W, pady=3)

        self.confusion_title = ttk.Label(
            self,
            text="Matriz de Confusão:",
            style=STYLE_SECTION,
        )
        self.confusion_title.pack(anchor=tk.W, pady=(8, 4))
        self.confusion = ConfusionMatrixView(self)
        self.confusion.pack(fill=tk.X)
        self.confusion.clear()

    def render(self, result):
        if isinstance(result.metrics, ClassificationMetrics):
            metrics = result.metrics

            self.configure(text=" 📊 Métricas (média macro) ")
            self.confusion_title.pack(anchor=tk.W, pady=(8, 4))
            self.confusion.pack(fill=tk.X)

            self.metric_labels["acc"].config(
                text=f"Acurácia: {metrics.accuracy*100:.2f}%"
            )
            self.metric_labels["prec"].config(
                text=f"Precisão: {metrics.precision*100:.2f}%"
            )
            self.metric_labels["rec"].config(text=f"Recall: {metrics.recall*100:.2f}%")
            self.metric_labels["f1"].config(text=f"F1-measure: {metrics.f1*100:.2f}%")
            self.confusion.render(metrics.confusion_matrix, result.class_names)
            return

        if isinstance(result.metrics, RegressionMetrics):
            metrics = result.metrics

            self.configure(text=" 📊 Métricas de Regressão ")
            self.confusion_title.pack_forget()
            self.confusion.pack_forget()

            self.metric_labels["acc"].config(text=f"MAE: {metrics.mae:.4f}")
            self.metric_labels["prec"].config(text=f"MSE: {metrics.mse:.4f}")
            self.metric_labels["rec"].config(text=f"RMSE: {metrics.rmse:.4f}")
            self.metric_labels["f1"].config(text=f"R²: {metrics.r2:.4f}")
            return

        self.configure(text=" 📊 Métricas (média macro) ")
        self.confusion_title.pack(anchor=tk.W, pady=(8, 4))
        self.confusion.pack(fill=tk.X)

        for key, name in METRIC_NAMES.items():
            self.metric_labels[key].config(text=f"{name}: -")
        self.confusion.clear()
