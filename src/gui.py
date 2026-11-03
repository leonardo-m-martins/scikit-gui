import ast
import math
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from src.enums import MethodResultField, Modes, Preprocessors
from src.loader import load_dataset
from src.ml_classification import CLASSIFICATION_CONFIG
from src.ml_common import ModelSpec, run_all_methods, run_method
from src.ml_regression import REGRESSION_CONFIG
from src.shared import (
    ClassificationMetrics,
    MethodResult,
    RegressionMetrics,
    format_metadata,
)

try:
    from sklearn.utils._param_validation import Hidden, StrOptions
except ImportError:
    Hidden = None
    StrOptions = None

DEFAULT_FILE = "./data/spambase.arff"

SORT_NEWEST = "Mais novo"
SORT_OLDEST = "Mais velho"
SORT_OPTIONS = (SORT_NEWEST, SORT_OLDEST)

PAGE_SIZE_OPTIONS = (50, 100, 200, 500)
DEFAULT_PAGE_SIZE = 100

SUMMARY_FIELDS = (
    "Dataset",
    "Modo",
    "Amostras",
    "Features",
    "Alvo",
    "Classes",
    "Tempo de carga",
)

MODE_CONFIGS = {
    Modes.CLASSIFICATION: CLASSIFICATION_CONFIG,
    Modes.REGRESSION: REGRESSION_CONFIG,
}


def method_values(mode: Modes) -> list[str]:
    return [method.value for method in MODE_CONFIGS[mode].specs]


def string_parameter_options(
    model,
    parameter_name,
) -> tuple[list[str], bool] | None:
    constraints = getattr(model, "_parameter_constraints", {}).get(
        parameter_name,
        (),
    )

    options = []
    allows_none = False
    has_other_types = False

    for constraint in constraints:
        if StrOptions is not None and isinstance(constraint, StrOptions):
            options.extend(constraint.options)
        elif constraint is None or constraint is type(None):
            allows_none = True
        elif Hidden is not None and isinstance(constraint, Hidden):
            continue
        else:
            has_other_types = True

    if not options:
        return None

    result = sorted(set(options))
    if allows_none:
        result.append("None")

    return result, not has_other_types


METADATA_FIELDS = tuple(
    field for field in MethodResultField if field is not MethodResultField.model_params
)

METRIC_NAMES = {
    "acc": "Acurácia",
    "prec": "Precisão",
    "rec": "Recall",
    "f1": "F1-measure",
}

STYLE_METADATA = "Metadata.TLabel"
STYLE_METRIC = "Metric.TLabel"
STYLE_SECTION = "Section.TLabel"
STYLE_MATRIX_HEADER = "MatrixHeader.TLabel"
STYLE_MATRIX_CELL = "MatrixCell.TLabel"


def configure_styles(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(STYLE_METADATA, font=("", 9, "bold"))
    style.configure(STYLE_METRIC, font=("", 10, "bold"))
    style.configure(STYLE_SECTION, font=("", 10, "bold"))
    style.configure(STYLE_MATRIX_HEADER, font=("", 9, "bold"))
    style.configure(STYLE_MATRIX_CELL, font=("", 9))


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.content = ttk.Frame(self.canvas)
        self.content_window = self.canvas.create_window(
            (0, 0), window=self.content, anchor=tk.NW
        )

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.content.bind("<Configure>", self._on_content_configure)
        self._wheel_bindings = []
        self._bind_mousewheel()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.content_window, width=event.width)
        self.canvas.after_idle(self._update_scrollbar)

    def _on_content_configure(self, _event):
        self.canvas.after_idle(self._update_scrollbar)

    def _update_scrollbar(self):
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        self.canvas.configure(scrollregion=bbox)
        available_height = self.canvas.winfo_height()
        if available_height <= 1:
            return
        content_height = bbox[3] - bbox[1]
        if content_height > available_height and not self.scrollbar.winfo_ismapped():
            self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        elif content_height <= available_height and self.scrollbar.winfo_ismapped():
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def _bind_mousewheel(self):
        top_level = self.winfo_toplevel()
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            funcid = top_level.bind(sequence, self._on_mousewheel, add="+")
            if funcid:
                self._wheel_bindings.append((top_level, sequence, funcid))

    def _on_destroy(self, event):
        if event.widget is not self:
            return
        for widget, sequence, funcid in self._wheel_bindings:
            widget.unbind(sequence, funcid)
        self._wheel_bindings.clear()

    def _on_mousewheel(self, event):
        if not self.scrollbar.winfo_ismapped():
            return
        widget = event.widget
        if not isinstance(widget, tk.Misc):
            return
        while widget is not None and widget is not self.canvas:
            widget = widget.master
        if widget is None:
            return
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
        return "break"


class ParameterDialog(tk.Toplevel):
    def __init__(self, parent, spec: ModelSpec, title: str):
        super().__init__(parent)
        self.spec = spec
        self.result: ModelSpec | None = None
        self.vars: dict[str, tk.StringVar] = {}

        self.title(title)
        self.transient(parent)
        self.geometry("560x640")
        self.minsize(420, 300)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build_widgets()
        self.grab_set()
        self.focus_set()

    def _build_widgets(self):
        scroll = ScrollableFrame(self)
        scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        content = scroll.content
        content.columnconfigure(1, weight=1)

        model = self.spec.model_type()
        defaults = model.get_params()
        values = {**defaults, **self.spec.params}

        for row, key in enumerate(sorted(values)):
            ttk.Label(content, text=key).grid(
                row=row,
                column=0,
                sticky=tk.W,
                padx=(0, 10),
                pady=3,
            )

            value = values[key]
            option_info = string_parameter_options(model, key)

            if option_info is not None:
                options, read_only = option_info
                use_combobox = (value is None or isinstance(value, str)) and (
                    value is None or value in options
                )
            else:
                options = []
                read_only = False
                use_combobox = False

            if use_combobox:
                var = tk.StringVar(
                    value="None" if value is None else value,
                )
                widget = ttk.Combobox(
                    content,
                    textvariable=var,
                    values=options,
                    state="readonly" if read_only else "normal",
                )
            else:
                var = tk.StringVar(value=repr(value))
                widget = ttk.Entry(content, textvariable=var)

            self.vars[key] = var
            widget.grid(
                row=row,
                column=1,
                sticky=tk.EW,
                pady=3,
            )

        action_frame = ttk.Frame(self)
        action_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        ttk.Button(
            action_frame,
            text="Cancelar",
            command=self._cancel,
        ).pack(side=tk.RIGHT)

        ttk.Button(
            action_frame,
            text="Aplicar",
            command=self._apply,
        ).pack(side=tk.RIGHT, padx=(0, 5))

    def _apply(self):
        params = {}

        for key, var in self.vars.items():
            text = var.get().strip()
            try:
                params[key] = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                params[key] = text

        try:
            self.spec.model_type(**params)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Invalid parameters",
                str(exc),
                parent=self,
            )
            return

        self.result = self.spec.update_params(params)
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class ResultsList(ttk.Frame):
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self._on_select = on_select
        self._results = []
        self.sort_var = tk.StringVar(value=SORT_NEWEST)

        self.sort_box = ttk.Combobox(
            self,
            textvariable=self.sort_var,
            values=SORT_OPTIONS,
            state="readonly",
        )
        self.sort_box.pack(fill=tk.X, pady=(0, 5))

        self.listbox = tk.Listbox(self, width=30)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._handle_select)
        self.sort_var.trace_add("write", self._handle_sort)

    def set_results(self, results):
        self._results = list(results)
        self._refresh()

    def selected_result(self):
        selection = self.listbox.curselection()
        if not selection:
            return None
        results = self._sorted_results()
        index = selection[0]
        if index >= len(results):
            return None
        return results[index]

    def _sorted_results(self):
        newest = self.sort_var.get() == SORT_NEWEST
        return sorted(
            self._results, key=lambda result: result.timestamp, reverse=newest
        )

    def _refresh(self):
        self.listbox.delete(0, tk.END)
        for result in self._sorted_results():
            self.listbox.insert(tk.END, result.get_display_name())

    def _handle_sort(self, *_args):
        self._refresh()

    def _handle_select(self, _event):
        result = self.selected_result()
        if result is not None:
            self._on_select(result)


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


class ResultsView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        left_pane = ttk.Frame(self)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.results_list = ResultsList(left_pane, on_select=self.show_result)
        self.results_list.pack(fill=tk.BOTH, expand=True)

        self.right_pane = ttk.LabelFrame(
            self, text=" 📈 Detalhes da Avaliação ", padding=15
        )
        self.right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_frame = ScrollableFrame(self.right_pane)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        content = scroll_frame.content

        self.details = DetailsPanel(content)
        self.details.pack(fill=tk.X)

        self.metrics = MetricsPanel(content)
        self.metrics.pack(fill=tk.X, pady=(15, 0))

    def refresh(self, results):
        self.results_list.set_results(results)

    def show_result(self, result):
        self.right_pane.config(text=f" 📈 {result.method_name} ")
        self.details.render(result)
        self.metrics.render(result)


def format_cell(value) -> str:
    if value is None:
        return "-"

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "?"
        return f"{float(value):.6g}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return str(value)


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


class DatasetTable(ttk.LabelFrame):
    def __init__(self, parent, on_page_change, on_page_size_change):
        super().__init__(parent, text=" 🔎 Amostras ", padding=10)
        self._on_page_change = on_page_change
        self._on_page_size_change = on_page_size_change
        self._page = 0
        self._pages = 1

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))

        ttk.Label(toolbar, text="Linhas por página:").pack(side=tk.LEFT)
        self.page_size_var = tk.StringVar(value=str(DEFAULT_PAGE_SIZE))
        self.page_size_box = ttk.Combobox(
            toolbar,
            textvariable=self.page_size_var,
            values=[str(size) for size in PAGE_SIZE_OPTIONS],
            state="readonly",
            width=5,
        )
        self.page_size_box.pack(side=tk.LEFT, padx=(5, 15))
        self.page_size_box.bind("<<ComboboxSelected>>", self._handle_page_size)

        self.first_button = ttk.Button(
            toolbar,
            text="⏮",
            width=3,
            command=lambda: self._request_page(0),
        )
        self.first_button.pack(side=tk.LEFT, padx=1)

        self.prev_button = ttk.Button(
            toolbar,
            text="◀",
            width=3,
            command=lambda: self._request_page(self._page - 1),
        )
        self.prev_button.pack(side=tk.LEFT, padx=1)

        self.next_button = ttk.Button(
            toolbar,
            text="▶",
            width=3,
            command=lambda: self._request_page(self._page + 1),
        )
        self.next_button.pack(side=tk.LEFT, padx=1)

        self.last_button = ttk.Button(
            toolbar,
            text="⏭",
            width=3,
            command=lambda: self._request_page(self._pages - 1),
        )
        self.last_button.pack(side=tk.LEFT, padx=1)

        self.page_var = tk.StringVar(value="Nenhum dado carregado.")
        ttk.Label(toolbar, textvariable=self.page_var).pack(
            side=tk.LEFT,
            padx=(15, 0),
        )

        self.tree = ttk.Treeview(self, show="headings", selectmode="browse")
        self.tree.grid(row=1, column=0, sticky="nsew")

        self.vscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.vscroll.grid(row=1, column=1, sticky=tk.NS)

        self.hscroll = ttk.Scrollbar(
            self, orient=tk.HORIZONTAL, command=self.tree.xview
        )
        self.hscroll.grid(row=2, column=0, sticky=tk.EW)

        self.tree.configure(
            yscrollcommand=self.vscroll.set,
            xscrollcommand=self.hscroll.set,
        )

        self.set_columns([], "")
        self.set_page_state(0, DEFAULT_PAGE_SIZE, 0)

    def set_columns(self, feature_names, target_name):
        self.tree.delete(*self.tree.get_children())

        names = list(feature_names or [])
        columns = ["row"]
        columns.extend(f"c{index}" for index in range(len(names)))
        columns.append("target")
        self.tree.configure(columns=columns)

        self._configure_column("row", "#", width=70, anchor=tk.E)
        for index, name in enumerate(names):
            self._configure_column(f"c{index}", name, anchor=tk.E)
        self._configure_column("target", target_name or "Alvo", anchor=tk.W)

    def _configure_column(self, column, heading, width=None, anchor=tk.W):
        if width is None:
            width = min(max(len(heading) * 8, 70), 200)
        self.tree.heading(column, text=heading, anchor=tk.W)
        self.tree.column(
            column,
            width=width,
            minwidth=50,
            stretch=False,
            anchor=anchor,
        )

    def clear_rows(self):
        self.tree.delete(*self.tree.get_children())

    def render_rows(self, dataset, start, end):
        self.tree.delete(*self.tree.get_children())

        for index in range(start, end):
            values = [index + 1]
            values.extend(format_cell(value) for value in dataset.features[index])
            values.append(format_cell(dataset.target[index]))
            self.tree.insert("", tk.END, values=values)

        self.tree.yview_moveto(0)
        self.tree.xview_moveto(0)

    def set_page_state(self, page, page_size, row_count):
        self._page = page
        self._pages = max(1, math.ceil(row_count / page_size)) if row_count else 1

        if row_count:
            start = page * page_size + 1
            end = min((page + 1) * page_size, row_count)
            label = f"Linhas {start}–{end} de {row_count:,}"
            self.page_var.set(label.replace(",", "."))
        else:
            self.page_var.set("Nenhum dado carregado.")

        self._set_enabled(self.first_button, page > 0)
        self._set_enabled(self.prev_button, page > 0)
        self._set_enabled(self.next_button, page < self._pages - 1)
        self._set_enabled(self.last_button, page < self._pages - 1)

    @staticmethod
    def _set_enabled(button: ttk.Button, enabled: bool):
        button.state(["!disabled"] if enabled else ["disabled"])

    def _request_page(self, page):
        if page != self._page:
            self._on_page_change(page)

    def _handle_page_size(self, _event=None):
        self._on_page_size_change(int(self.page_size_var.get()))


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


class DataView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.dataset = None
        self._page = 0
        self._page_size = DEFAULT_PAGE_SIZE
        self._row_count = 0

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.file_var = tk.StringVar(value="Nenhum dataset carregado.")
        ttk.Label(self, textvariable=self.file_var, style=STYLE_SECTION).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=15,
            pady=(15, 5),
        )

        self.table = DatasetTable(
            self,
            on_page_change=self._on_page_change,
            on_page_size_change=self._on_page_size_change,
        )
        self.table.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def render(self, dataset):
        self.dataset = dataset
        self._row_count = len(dataset.target) if dataset is not None else 0
        self._page = 0

        self.file_var.set(
            f"📁 {dataset.name}" if dataset is not None else "Nenhum dataset carregado."
        )
        self.table.set_columns(
            dataset.feature_names if dataset is not None else [],
            dataset.target_name if dataset is not None else "",
        )
        self._render_page()

    def _on_page_change(self, page):
        self._page = page
        self._render_page()

    def _on_page_size_change(self, page_size):
        self._page_size = page_size
        self._page = 0
        self._render_page()

    def _render_page(self):
        if self.dataset is None or self._row_count == 0:
            self.table.clear_rows()
            self.table.set_page_state(0, self._page_size, 0)
            return

        pages = max(1, math.ceil(self._row_count / self._page_size))
        self._page = min(max(self._page, 0), pages - 1)

        start = self._page * self._page_size
        end = min(start + self._page_size, self._row_count)
        self.table.render_rows(self.dataset, start, end)
        self.table.set_page_state(self._page, self._page_size, self._row_count)


class ConfigPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        on_run_selected,
        on_run_all,
        on_dataset_loaded=None,
    ):
        super().__init__(parent, text=" ⚙️ Configurações do Modelo ", padding=20)
        self._on_run_selected = on_run_selected
        self._on_run_all = on_run_all
        self._on_dataset_loaded = on_dataset_loaded

        self.path_var = tk.StringVar(value=DEFAULT_FILE)
        self.mode_var = tk.StringVar(value=Modes.CLASSIFICATION.value)
        self.method_var = tk.StringVar()
        self.preprocessing_var = tk.StringVar(value=Preprocessors.NONE.value)
        self.split_var = tk.IntVar(value=20)
        self.dataset = None
        self._loaded_mode = Modes.CLASSIFICATION
        self._running = False
        self._progress_total = 0
        self._progress_index = 0
        self._progress_job = None
        self._current_method = ""
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="📁 Arquivo de Dados:").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10), padx=(0, 10)
        )
        self.path_entry = ttk.Entry(self, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 10), padx=(0, 10))
        self.browse_button = ttk.Button(
            self,
            text="🔍 Procurar",
            command=self._browse,
        )
        self.browse_button.grid(row=0, column=2, pady=(0, 10))

        ttk.Label(self, text="🎚️ Modo:").grid(
            row=1, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.mode_box = ttk.Combobox(
            self,
            textvariable=self.mode_var,
            values=Modes.values(),
            state="readonly",
        )
        self.mode_box.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(self, text="🛠️ Método:").grid(
            row=2, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.method_box = ttk.Combobox(
            self,
            textvariable=self.method_var,
            values=method_values(Modes.CLASSIFICATION),
            state="readonly",
        )
        self.method_box.grid(
            row=2,
            column=1,
            sticky=tk.EW,
            pady=10,
        )
        self.method_box.bind("<<ComboboxSelected>>", self._on_method_changed)

        self.params_button = ttk.Button(
            self,
            text="⚙️ Parâmetros",
            command=self._open_params,
        )
        self.params_button.grid(
            row=2,
            column=2,
            padx=(10, 0),
            pady=10,
        )

        ttk.Label(self, text="🗒️ Pré-processamento:").grid(
            row=3, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.preprocessing_box = ttk.Combobox(
            self,
            textvariable=self.preprocessing_var,
            values=Preprocessors.values(),
            state=tk.DISABLED,
        )
        self.preprocessing_box.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )
        self.preprocessing_box.bind(
            "<<ComboboxSelected>>",
            self._on_preprocessing_changed,
        )

        ttk.Label(self, text="📊 % de Teste:").grid(
            row=4, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.split_box = ttk.Spinbox(
            self,
            textvariable=self.split_var,
            from_=0,
            to=100,
            increment=1,
        )
        self.split_box.grid(
            row=4,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )

        action_frame = ttk.Frame(self)
        action_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(25, 0))

        self.run_selected_button = ttk.Button(
            action_frame,
            text="▶️ Executar Selecionado",
            command=self._on_run_selected,
        )
        self.run_selected_button.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 5),
        )

        self.run_all_button = ttk.Button(
            action_frame,
            text="🚀 Executar Todos",
            command=self._on_run_all,
        )
        self.run_all_button.pack(
            side=tk.RIGHT,
            fill=tk.X,
            expand=True,
            padx=(5, 0),
        )

        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(
            row=6,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(10, 0),
        )

        self.spinner = ttk.Progressbar(
            self.status_frame,
            mode="indeterminate",
            length=100,
        )
        self.spinner.pack(side=tk.LEFT, padx=(0, 8))

        self.current_method_var = tk.StringVar()
        ttk.Label(
            self.status_frame,
            textvariable=self.current_method_var,
        ).pack(side=tk.LEFT)

        self.progress_var = tk.StringVar()
        ttk.Label(
            self.status_frame,
            textvariable=self.progress_var,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.status_frame.grid_remove()

    def set_running(
        self,
        running: bool,
        total: int = 1,
        method_name: str = "",
    ):
        self._running = running

        if running:
            self._set_controls_enabled(False)
            self._progress_total = total
            self._progress_index = 1
            self._current_method = method_name
            self.status_frame.grid()
            self.spinner.start(12)
            self._update_progress_label()
            self._schedule_progress_tick()
            return

        self._cancel_progress_tick()
        self.spinner.stop()
        self.status_frame.grid_remove()
        self.current_method_var.set("")
        self.progress_var.set("")
        self._set_controls_enabled(True)

    def set_current_method(self, method_name: str):
        self._current_method = method_name
        self._update_progress_label()

    def mark_method_completed(self):
        if not self._running:
            return

        self._progress_total = max(self._progress_total - 1, 1)
        self._progress_index = 1
        self._cancel_progress_tick()
        self._update_progress_label()

        if self._progress_total > 1:
            self._schedule_progress_tick()

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        self.path_entry.configure(state=state)
        self.browse_button.configure(state=state)
        self.split_box.configure(state=state)
        self.run_selected_button.configure(state=state)
        self.run_all_button.configure(state=state)
        self.params_button.configure(state=state)

        combo_state = "readonly" if enabled else tk.DISABLED
        self.mode_box.configure(state=combo_state)
        self.method_box.configure(state=combo_state)

        preprocessing_state = (
            "readonly" if enabled and self.method_var.get() else tk.DISABLED
        )
        self.preprocessing_box.configure(state=preprocessing_state)

    def _update_progress_label(self):
        self.current_method_var.set(self._current_method)

        if not self._running or self._progress_total <= 1:
            self.progress_var.set("")
            return

        self.progress_var.set(f"{self._progress_index}/{self._progress_total}")

    def _schedule_progress_tick(self):
        if self._progress_job is not None:
            return

        if not self._running or self._progress_total <= 1:
            return

        self._progress_job = self.after(2000, self._tick_progress)

    def _tick_progress(self):
        self._progress_job = None

        if not self._running or self._progress_total <= 1:
            return

        self._progress_index = min(
            self._progress_index + 1,
            self._progress_total,
        )
        self._update_progress_label()

        if self._progress_index < self._progress_total:
            self._schedule_progress_tick()

    def _cancel_progress_tick(self):
        if self._progress_job is not None:
            self.after_cancel(self._progress_job)
            self._progress_job = None

    def get_values(self):
        return (
            self.dataset,
            int(self.split_var.get()),
            Modes(self.mode_var.get()),
            self.method_var.get(),
        )

    def _open_params(self):
        method = self.method_var.get()
        if not method:
            messagebox.showwarning("Parameters", "Select a method first.")
            return

        mode = Modes(self.mode_var.get())
        spec = MODE_CONFIGS[mode].specs.get(method)
        if spec is None:
            messagebox.showwarning("Parameters", "Method is not supported.")
            return

        dialog = ParameterDialog(
            self,
            spec,
            title=f"Parâmetros — {method}",
        )
        self.wait_window(dialog)

        if dialog.result is not None:
            MODE_CONFIGS[mode].specs[method] = dialog.result

    def _browse(self):
        if filename := filedialog.askopenfilename():
            self.path_var.set(filename)
            self._update_dataset()

    def _on_mode_changed(self, _event=None):
        previous_mode = self._loaded_mode
        self._update_method_values()

        if not self._update_dataset():
            self.mode_var.set(previous_mode.value)
            self._update_method_values()

    def _update_method_values(self):
        mode = Modes(self.mode_var.get())
        self.method_box.configure(values=method_values(mode))
        self.method_var.set("")
        self._on_method_changed()

    def _on_method_changed(self, _event=None):
        method = self.method_var.get()
        if not method:
            self.preprocessing_var.set(Preprocessors.NONE.value)
            self.preprocessing_box.configure(state=tk.DISABLED)
            return

        mode = Modes(self.mode_var.get())
        spec = MODE_CONFIGS[mode].specs.get(method)
        if spec is None:
            self.preprocessing_var.set(Preprocessors.NONE.value)
            self.preprocessing_box.configure(state=tk.DISABLED)
            return

        self.preprocessing_var.set(spec.preprocessing.value)
        self.preprocessing_box.configure(state="readonly")

    def _on_preprocessing_changed(self, _event=None):
        method = self.method_var.get()
        if not method:
            return

        mode = Modes(self.mode_var.get())
        spec = MODE_CONFIGS[mode].specs.get(method)
        if spec is None:
            return

        preprocessing = Preprocessors(self.preprocessing_var.get())
        MODE_CONFIGS[mode].specs[method] = spec.update_params(
            {},
            preprocessing=preprocessing,
        )

    def reload_dataset(self) -> bool:
        return self._update_dataset()

    def _update_dataset(self) -> bool:
        mode = Modes(self.mode_var.get())

        try:
            dataset = load_dataset(self.path_var.get(), mode=mode)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return False

        self.dataset = dataset
        self._loaded_mode = mode

        if self._on_dataset_loaded is not None:
            self._on_dataset_loaded(dataset, mode)

        return True


class App:
    def __init__(self, root):
        self.root = root
        self.results: list[MethodResult] = []
        self._busy = False
        self._worker_queue = queue.Queue()
        configure_styles(root)

        self.config = ConfigPanel(
            root,
            on_run_selected=self.run_selected,
            on_run_all=self.run_all,
            on_dataset_loaded=self._on_dataset_loaded,
        )
        self.config.pack(fill=tk.X, padx=20, pady=(20, 10))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.summary_view = DataSummaryView(self.notebook)
        self.data_view = DataView(self.notebook)
        self.results_view = ResultsView(self.notebook)
        self.notebook.add(self.summary_view, text=" 📄 Resumo Dados ")
        self.notebook.add(self.data_view, text=" 📋 Dados ")
        self.notebook.add(self.results_view, text=" 📈 Resultados ")

        self.config.reload_dataset()

    def _on_dataset_loaded(self, dataset, mode):
        self.summary_view.render(dataset, mode)
        self.data_view.render(dataset)

    def run_selected(self):
        dataset, split, mode, method = self.config.get_values()
        mode_config = MODE_CONFIGS[mode]

        if not method:
            messagebox.showwarning("Method", "Select a method first.")
            return

        def work():
            return run_method(
                method,
                dataset,
                split,
                mode_config,
                stratify=mode is Modes.CLASSIFICATION,
            )

        self._run_in_background(
            work,
            self._append_result,
            total=1,
            method_name=method,
        )

    def run_all(self):
        dataset, split, mode, _ = self.config.get_values()
        mode_config = MODE_CONFIGS[mode]
        methods = [method.value for method in mode_config.specs]

        def work():
            return run_all_methods(
                dataset,
                split,
                mode_config,
                stratify=mode is Modes.CLASSIFICATION,
                on_method_start=lambda name: self._worker_queue.put(
                    ("method_started", name)
                ),
                on_method_complete=lambda: self._worker_queue.put(
                    ("method_completed", None)
                ),
            )

        self._run_in_background(
            work,
            self._append_results,
            total=len(methods),
            method_name=methods[0] if methods else "",
        )

    def _run_in_background(
        self,
        work,
        on_success,
        *,
        total: int = 1,
        method_name: str = "",
    ):
        if self._busy:
            return

        self._clear_worker_queue()
        self._busy = True
        self.config.set_running(True, total=total, method_name=method_name)

        def worker():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                self._worker_queue.put(("error", exc))
            else:
                self._worker_queue.put(("success", result))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_worker, on_success)

    def _poll_worker(self, on_success):
        try:
            kind, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker, on_success)
            return

        if kind == "method_started":
            self.config.set_current_method(payload)
            self.root.after(100, self._poll_worker, on_success)
            return

        if kind == "method_completed":
            self.config.mark_method_completed()
            self.root.after(100, self._poll_worker, on_success)
            return

        self._busy = False
        self.config.set_running(False)

        if kind == "error":
            messagebox.showerror("Execution error", str(payload))
            return

        on_success(payload)

    def _clear_worker_queue(self):
        while True:
            try:
                self._worker_queue.get_nowait()
            except queue.Empty:
                return

    def _append_result(self, result):
        if result is None:
            return
        self.results.append(result)
        self.results_view.refresh(self.results)

    def _append_results(self, results):
        self.results.extend(results)
        self.results_view.refresh(self.results)


def main():
    root = tk.Tk()
    root.title("📊 Classificação e Regressão")
    root.geometry("1000x800")
    root.minsize(820, 660)

    App(root)

    root.mainloop()


if __name__ == "__main__":
    main()
