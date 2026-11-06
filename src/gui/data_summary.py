"""Dataset summary, statistics, and class distribution views."""

import tkinter as tk
from tkinter import ttk

import numpy as np

from src.enums import MethodResultField, Modes
from src.gui.styles import STYLE_METADATA
from src.shared import format_metadata

SUMMARY_FIELDS = (
    "Dataset",
    "Modo",
    "Amostras",
    "Features",
    "Alvo",
    "Classes",
    "Tempo de carga",
)


def format_number(value: float) -> str:
    return f"{value:.4g}"


class SummaryBar(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text=" 📄 Resumo do Dataset ", padding=10)
        self.labels: dict[str, ttk.Label] = {}

        for index, field in enumerate(SUMMARY_FIELDS):
            row, column = divmod(index, 4)
            ttk.Label(self, text=f"{field}:", style=STYLE_METADATA).grid(
                row=row,
                column=column * 2,
                sticky=tk.W,
                padx=(0, 5),
                pady=2,
            )
            label = ttk.Label(self, text="-")
            label.grid(
                row=row,
                column=column * 2 + 1,
                sticky=tk.W,
                padx=(0, 20),
                pady=2,
            )
            self.labels[field] = label

    def render(self, dataset, mode):
        if dataset is None:
            for label in self.labels.values():
                label.config(text="-")
            return

        n_features = len(dataset.feature_names)
        if not n_features and len(dataset.features):
            n_features = len(dataset.features[0])

        n_samples = len(dataset.target)
        self.labels["Dataset"].config(text=dataset.name)
        self.labels["Modo"].config(text=mode.value)
        self.labels["Amostras"].config(text=f"{n_samples:,}".replace(",", "."))
        self.labels["Features"].config(text=str(n_features))
        self.labels["Alvo"].config(text=dataset.target_name or "—")
        self.labels["Classes"].config(
            text=str(len(dataset.target_names)) if dataset.target_names else "—"
        )
        self.labels["Tempo de carga"].config(
            text=format_metadata(MethodResultField.load_time, dataset.load_time)
        )


class StatsView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        headings = {
            "column": ("Coluna", 240, tk.W),
            "mean": ("Média", 110, tk.E),
            "std": ("Desvio", 110, tk.E),
            "min": ("Mín", 110, tk.E),
            "max": ("Máx", 110, tk.E),
            "nulls": ("Nulos", 80, tk.E),
        }

        self.tree = ttk.Treeview(
            self,
            columns=tuple(headings),
            show="headings",
            selectmode="browse",
        )
        for column, (heading, width, anchor) in headings.items():
            self.tree.heading(column, text=heading, anchor=tk.W)
            self.tree.column(
                column,
                width=width,
                minwidth=60,
                stretch=column == "column",
                anchor=anchor,
            )

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

    def render(self, dataset, mode):
        self.tree.delete(*self.tree.get_children())
        if dataset is None or len(dataset.target) == 0:
            return

        features = np.asarray(dataset.features, dtype=float)
        if features.ndim != 2 or features.shape[1] == 0:
            return

        names = list(dataset.feature_names)
        if len(names) != features.shape[1]:
            names = [f"Feature {index + 1}" for index in range(features.shape[1])]

        columns = [(name, features[:, index]) for index, name in enumerate(names)]

        if mode is Modes.REGRESSION:
            target = np.asarray(dataset.target, dtype=float)
            columns.append((dataset.target_name or "Alvo", target))

        for name, values in columns:
            valid = values[~np.isnan(values)]
            if valid.size:
                mean = format_number(valid.mean())
                std = format_number(valid.std())
                minimum = format_number(valid.min())
                maximum = format_number(valid.max())
            else:
                mean = std = minimum = maximum = "—"

            self.tree.insert(
                "",
                tk.END,
                values=(
                    name,
                    mean,
                    std,
                    minimum,
                    maximum,
                    int(np.isnan(values).sum()),
                ),
            )


class ClassDistributionView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        headings = {
            "class": ("Classe", 240, tk.W),
            "count": ("Contagem", 120, tk.E),
            "percent": ("Percentual", 120, tk.E),
        }

        self.tree = ttk.Treeview(
            self,
            columns=tuple(headings),
            show="headings",
            selectmode="browse",
        )
        for column, (heading, width, anchor) in headings.items():
            self.tree.heading(column, text=heading, anchor=tk.W)
            self.tree.column(
                column,
                width=width,
                minwidth=60,
                stretch=column == "class",
                anchor=anchor,
            )

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

    def render(self, dataset, mode):
        self.tree.delete(*self.tree.get_children())
        if dataset is None or mode is not Modes.CLASSIFICATION:
            return

        values = np.asarray(dataset.target)
        unique, counts = np.unique(values, return_counts=True)
        totals = dict(zip(unique.tolist(), counts.tolist()))

        names = [name for name in dataset.target_names or [] if name in totals]
        if not names:
            names = sorted(totals)

        total = len(values)
        for name in names:
            count = totals.get(name, 0)
            percent = count / total * 100 if total else 0.0
            self.tree.insert(
                "",
                tk.END,
                values=(name, count, f"{percent:.2f}%"),
            )


class DataSummaryView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.dataset = None
        self.mode = Modes.CLASSIFICATION
        self._classes_visible = True

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.summary = SummaryBar(self)
        self.summary.grid(row=0, column=0, sticky=tk.EW, padx=15, pady=(15, 5))

        self.details = ttk.Notebook(self)
        self.stats = StatsView(self.details)
        self.classes = ClassDistributionView(self.details)
        self.details.add(self.stats, text=" 📈 Estatísticas ")
        self.details.add(self.classes, text=" 🏷️ Classes ")
        self.details.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def render(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode if isinstance(mode, Modes) else Modes(mode)

        self.summary.render(dataset, self.mode)
        self._update_classes_tab()
        self.stats.render(dataset, self.mode)
        self.classes.render(dataset, self.mode)

    def _update_classes_tab(self):
        visible = self.mode is Modes.CLASSIFICATION
        if visible and not self._classes_visible:
            self.details.add(self.classes, text=" 🏷️ Classes ")
            self._classes_visible = True
        elif not visible and self._classes_visible:
            self.details.hide(self.classes)
            self._classes_visible = False
