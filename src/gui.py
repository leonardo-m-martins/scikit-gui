import tkinter as tk
from tkinter import filedialog, ttk

from src.enums import Methods
from src.ml_clas_sk import run_all_methods, run_individual_method
from src.shared import ClassificationMetrics, MethodResult, format_duration

DEFAULT_FILE = "./data/spambase.arff"

SORT_NEWEST = "Mais novo"
SORT_OLDEST = "Mais velho"
SORT_OPTIONS = (SORT_NEWEST, SORT_OLDEST)

METADATA_FIELDS = (
    ("method", "Método:"),
    ("timestamp", "Execução:"),
    ("dataset", "Dataset:"),
    ("samples", "Amostras:"),
    ("features", "Features:"),
    ("classes", "Classes:"),
    ("split", "Divisão de teste:"),
    ("train_test", "Treino / Teste:"),
    ("load_time", "Tempo de carga:"),
    ("train_time", "Tempo de treino:"),
    ("test_time", "Tempo de teste:"),
    ("total_time", "Tempo total:"),
    ("random_state", "Random state:"),
    ("preprocessing", "Pré-processamento:"),
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

        self.metadata = {}
        for index, (key, text) in enumerate(METADATA_FIELDS):
            row, column = divmod(index, 2)
            ttk.Label(self, text=text, style=STYLE_METADATA).grid(
                row=row, column=column * 2, sticky=tk.W, padx=(0, 5), pady=2
            )
            value = ttk.Label(self, text="-")
            value.grid(
                row=row, column=column * 2 + 1, sticky=tk.W, padx=(0, 20), pady=2
            )
            self.metadata[key] = value

        params_row = (len(METADATA_FIELDS) + 1) // 2
        ttk.Label(self, text="Hiperparâmetros:", style=STYLE_METADATA).grid(
            row=params_row, column=0, sticky=tk.NW, pady=2
        )
        self.params_value = ttk.Label(self, text="-", wraplength=520, justify=tk.LEFT)
        self.params_value.grid(
            row=params_row, column=1, columnspan=3, sticky=tk.W, pady=2
        )

    def render(self, result):
        self.metadata["method"].config(text=result.method_name)
        self.metadata["timestamp"].config(
            text=result.timestamp.strftime("%d/%m/%Y %H:%M:%S")
        )
        self.metadata["dataset"].config(text=result.dataset_name or "-")
        self.metadata["samples"].config(
            text="-" if result.n_samples is None else str(result.n_samples)
        )
        self.metadata["features"].config(
            text="-" if result.n_features is None else str(result.n_features)
        )
        self.metadata["classes"].config(
            text="-" if result.n_classes is None else str(result.n_classes)
        )
        self.metadata["split"].config(
            text=f"{result.test_split:g}%" if result.test_split is not None else "-"
        )
        self.metadata["train_test"].config(
            text=(
                f"{result.n_train} / {result.n_test}"
                if result.n_train is not None and result.n_test is not None
                else "-"
            )
        )
        self.metadata["load_time"].config(text=format_duration(result.load_time))
        self.metadata["train_time"].config(text=format_duration(result.train_time))
        self.metadata["test_time"].config(text=format_duration(result.test_time))
        self.metadata["total_time"].config(text=format_duration(result.total_time))
        self.metadata["random_state"].config(
            text="-" if result.random_state is None else str(result.random_state)
        )
        self.metadata["preprocessing"].config(text=result.preprocessing or "-")
        self.params_value.config(
            text=", ".join(
                f"{key}={value}" for key, value in result.model_params.items()
            )
            or "-"
        )


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

        ttk.Label(self, text="Matriz de Confusão:", style=STYLE_SECTION).pack(
            anchor=tk.W, pady=(8, 4)
        )
        self.confusion = ConfusionMatrixView(self)
        self.confusion.pack(fill=tk.X)
        self.confusion.clear()

    def render(self, result):
        if isinstance(result.metrics, ClassificationMetrics):
            metrics = result.metrics
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
        self.method_var = tk.StringVar()
        self.split_var = tk.IntVar(value=20)
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="📁 Arquivo de Dados:").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10), padx=(0, 10)
        )
        ttk.Entry(self, textvariable=self.path_var).grid(
            row=0, column=1, sticky=tk.EW, pady=(0, 10), padx=(0, 10)
        )
        ttk.Button(self, text="🔍 Procurar", command=self._browse).grid(
            row=0, column=2, pady=(0, 10)
        )

        ttk.Label(self, text="🛠️ Método:").grid(
            row=1, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        ttk.Combobox(
            self,
            textvariable=self.method_var,
            values=Methods.values(),
            state="readonly",
        ).grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=10)

        ttk.Label(self, text="📊 % de Teste:").grid(
            row=2, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        ttk.Spinbox(
            self,
            textvariable=self.split_var,
            from_=0,
            to=100,
            increment=1,
        ).grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=10)

        action_frame = ttk.Frame(self)
        action_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(25, 0))

        ttk.Button(
            action_frame,
            text="▶️ Executar Selecionado",
            command=self._on_run_selected,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(
            action_frame,
            text="🚀 Executar Todos",
            command=self._on_run_all,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    def get_values(self):
        return self.path_var.get(), int(self.split_var.get()), self.method_var.get()

    def _browse(self):
        if filename := filedialog.askopenfilename():
            self.path_var.set(filename)


class App:
    def __init__(self, root):
        self.root = root
        self.results: list[MethodResult] = []
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
        path, split, method = self.config.get_values()
        run_individual_method(path, split, method, self.results)
        self.results_view.refresh(self.results)

    def run_all(self):
        path, split, _ = self.config.get_values()
        run_all_methods(path, split, self.results)
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
