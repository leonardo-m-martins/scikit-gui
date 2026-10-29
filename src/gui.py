import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.enums import MethodResultField, Modes
from src.loader import load_dataset
from src.ml_classification import CLASSIFICATION_CONFIG
from src.ml_common import run_all_methods, run_method
from src.ml_regression import REGRESSION_CONFIG
from src.shared import (
    ClassificationMetrics,
    MethodResult,
    RegressionMetrics,
    format_metadata,
)

DEFAULT_FILE = "./data/spambase.arff"

SORT_NEWEST = "Mais novo"
SORT_OLDEST = "Mais velho"
SORT_OPTIONS = (SORT_NEWEST, SORT_OLDEST)

MODE_CONFIGS = {
    Modes.CLASSIFICATION: CLASSIFICATION_CONFIG,
    Modes.REGRESSION: REGRESSION_CONFIG,
}


def method_values(mode: Modes) -> list[str]:
    return [method.value for method in MODE_CONFIGS[mode].specs]


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


class ConfigPanel(ttk.LabelFrame):
    def __init__(self, parent, on_run_selected, on_run_all):
        super().__init__(parent, text=" ⚙️ Configurações do Modelo ", padding=20)
        self._on_run_selected = on_run_selected
        self._on_run_all = on_run_all

        self.path_var = tk.StringVar(value=DEFAULT_FILE)
        self.mode_var = tk.StringVar(value=Modes.CLASSIFICATION.value)
        self.method_var = tk.StringVar()
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
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )

        ttk.Label(self, text="📊 % de Teste:").grid(
            row=3, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.split_box = ttk.Spinbox(
            self,
            textvariable=self.split_var,
            from_=0,
            to=100,
            increment=1,
        )
        self.split_box.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )

        action_frame = ttk.Frame(self)
        action_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(25, 0))

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
            row=5,
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
        self._update_dataset()

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

        combo_state = "readonly" if enabled else tk.DISABLED
        self.mode_box.configure(state=combo_state)
        self.method_box.configure(state=combo_state)

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

    def _update_dataset(self) -> bool:
        mode = Modes(self.mode_var.get())

        try:
            dataset = load_dataset(self.path_var.get(), mode=mode)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return False

        self.dataset = dataset
        self._loaded_mode = mode
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
        )
        self.config.pack(fill=tk.X, padx=20, pady=20)

        self.results_view = ResultsView(root)
        self.results_view.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

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
    root.geometry("860x760")
    root.minsize(700, 620)

    App(root)

    root.mainloop()


if __name__ == "__main__":
    main()
