import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Any
from src.enums import Methods
from src.ml_clas_sk import run_individual_method, run_all_methods
from src.shared import MethodResult, Metrics, ClassificationMetrics, format_duration

DEFAULT_FILE = "./data/spambase.arff"
METHOD_RESULTS: List[MethodResult] = []


def open_file(path_entry):
    if filename := filedialog.askopenfilename():
        path_entry.delete(0, tk.END)
        path_entry.insert(0, filename)


def top_menu(parent):
    frm = ttk.LabelFrame(parent, text=" ⚙️ Configurações do Modelo ", padding=20)
    frm.pack(fill=tk.X, padx=20, pady=20)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text="📁 Arquivo de Dados:").grid(
        row=0, column=0, sticky=tk.W, pady=(0, 10), padx=(0, 10)
    )
    path_entry = ttk.Entry(frm)
    path_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 10), padx=(0, 10))
    path_entry.delete(0, tk.END)
    path_entry.insert(0, DEFAULT_FILE)
    ttk.Button(frm, text="🔍 Procurar", command=lambda: open_file(path_entry)).grid(
        row=0, column=2, pady=(0, 10)
    )

    ttk.Label(frm, text="🛠️ Método:").grid(
        row=1, column=0, sticky=tk.W, pady=10, padx=(0, 10)
    )
    method_box = ttk.Combobox(frm, values=Methods.values(), state="readonly")
    method_box.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=10)

    ttk.Label(frm, text="📊 % de Teste:").grid(
        row=2, column=0, sticky=tk.W, pady=10, padx=(0, 10)
    )
    percentage_box = ttk.Spinbox(frm, from_=0, to=100, increment=1)
    percentage_box.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=10)
    percentage_box.set(20)

    action_frame = ttk.Frame(frm)
    action_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(25, 0))

    # Trigger ML methods and notify the parent window to update the results UI
    def execute_single():
        run_individual_method(
            path_entry.get(),
            int(percentage_box.get()),
            method_box.get(),
            METHOD_RESULTS,
        )
        parent.event_generate("<<UpdateResults>>")

    def execute_all():
        run_all_methods(path_entry.get(), int(percentage_box.get()), METHOD_RESULTS)
        parent.event_generate("<<UpdateResults>>")

    ttk.Button(
        action_frame,
        text="▶️ Executar Selecionado",
        command=execute_single,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    ttk.Button(
        action_frame,
        text="🚀 Executar Todos",
        command=execute_all,
    ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))


def results_menu(parent):
    container = ttk.Frame(parent)
    container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    left_pane = ttk.Frame(container)
    left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

    sort_var = tk.StringVar(value="Mais novo")
    ttk.Combobox(
        left_pane,
        textvariable=sort_var,
        values=["Mais novo", "Mais velho"],
        state="readonly",
    ).pack(fill=tk.X, pady=(0, 5))

    listbox = tk.Listbox(left_pane, width=30)
    listbox.pack(fill=tk.BOTH, expand=True)

    right_pane = ttk.LabelFrame(
        container, text=" 📈 Detalhes da Avaliação ", padding=15
    )
    right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scroll_container = ttk.Frame(right_pane)
    scroll_container.pack(fill=tk.BOTH, expand=True)
    scroll_container.rowconfigure(0, weight=1)
    scroll_container.columnconfigure(0, weight=1)

    canvas = tk.Canvas(scroll_container, highlightthickness=0, borderwidth=0)
    canvas.grid(row=0, column=0, sticky="nsew")

    scrollbar = ttk.Scrollbar(
        scroll_container, orient=tk.VERTICAL, command=canvas.yview
    )
    canvas.configure(yscrollcommand=scrollbar.set)

    content = ttk.Frame(canvas)
    content_window = canvas.create_window((0, 0), window=content, anchor=tk.NW)

    details = ttk.Frame(content)
    details.pack(fill=tk.X)
    details.columnconfigure(1, weight=1)
    details.columnconfigure(3, weight=1)

    metadata_fields = [
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
    ]

    metadata = {}
    for index, (key, text) in enumerate(metadata_fields):
        row, col = divmod(index, 2)
        ttk.Label(details, text=text, font=("", 9, "bold")).grid(
            row=row, column=col * 2, sticky=tk.W, padx=(0, 5), pady=2
        )
        value = ttk.Label(details, text="-")
        value.grid(row=row, column=col * 2 + 1, sticky=tk.W, padx=(0, 20), pady=2)
        metadata[key] = value

    params_row = (len(metadata_fields) + 1) // 2
    ttk.Label(details, text="Hiperparâmetros:", font=("", 9, "bold")).grid(
        row=params_row, column=0, sticky=tk.NW, pady=2
    )
    params_value = ttk.Label(details, text="-", wraplength=520, justify=tk.LEFT)
    params_value.grid(row=params_row, column=1, columnspan=3, sticky=tk.W, pady=2)

    metrics_frame = ttk.LabelFrame(
        content, text=" 📊 Métricas (média macro) ", padding=10
    )
    metrics_frame.pack(fill=tk.X, pady=(15, 0))

    metric_names = {
        "acc": "Acurácia",
        "prec": "Precisão",
        "rec": "Recall",
        "f1": "F1-measure",
    }
    metric_labels = {
        key: ttk.Label(metrics_frame, text=f"{name}: -", font=("", 10, "bold"))
        for key, name in metric_names.items()
    }
    for lbl in metric_labels.values():
        lbl.pack(anchor=tk.W, pady=3)

    ttk.Label(metrics_frame, text="Matriz de Confusão:", font=("", 10, "bold")).pack(
        anchor=tk.W, pady=(8, 4)
    )
    cm_container = ttk.Frame(metrics_frame)
    cm_container.pack(fill=tk.X)

    def update_scrollbar():
        bbox = canvas.bbox("all")
        if bbox is None:
            return
        canvas.configure(scrollregion=bbox)
        available_height = canvas.winfo_height()
        if available_height <= 1:
            return
        content_height = bbox[3] - bbox[1]
        if content_height > available_height and not scrollbar.winfo_ismapped():
            scrollbar.grid(row=0, column=1, sticky=tk.NS)
        elif content_height <= available_height and scrollbar.winfo_ismapped():
            scrollbar.grid_remove()
            canvas.yview_moveto(0)

    def on_canvas_configure(event):
        canvas.itemconfigure(content_window, width=event.width)
        canvas.after_idle(update_scrollbar)

    def on_mousewheel(event):
        if not scrollbar.winfo_ismapped():
            return
        widget = event.widget
        if not isinstance(widget, tk.Misc):
            return
        while widget is not None and widget is not canvas:
            widget = widget.master
        if widget is None:
            return
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")
        else:
            canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
        return "break"

    canvas.bind("<Configure>", on_canvas_configure)
    content.bind("<Configure>", lambda _event: canvas.after_idle(update_scrollbar))
    top_level = parent.winfo_toplevel()
    top_level.bind_all("<MouseWheel>", on_mousewheel, add="+")
    top_level.bind_all("<Button-4>", on_mousewheel, add="+")
    top_level.bind_all("<Button-5>", on_mousewheel, add="+")

    def clear_confusion_matrix():
        for child in cm_container.winfo_children():
            child.destroy()
        ttk.Label(cm_container, text="-").grid(row=0, column=0, sticky=tk.W)

    def render_confusion_matrix(matrix, class_names):
        for child in cm_container.winfo_children():
            child.destroy()

        rows = [list(row) for row in matrix]
        size = len(rows)
        if not size:
            clear_confusion_matrix()
            return

        names = list(class_names or [])
        if len(names) != size:
            names = [f"Classe {index}" for index in range(size)]

        def make_cell(text, row, column, bold=False):
            cell = ttk.Label(
                cm_container,
                text=text,
                relief="solid",
                borderwidth=1,
                padding=(8, 4),
                anchor=tk.CENTER,
                justify=tk.CENTER,
                font=("", 9, "bold" if bold else "normal"),
            )
            cell.grid(row=row, column=column, sticky="nsew")

        make_cell("Real \\ Previsto", 0, 0, bold=True)
        for column, name in enumerate(names, start=1):
            make_cell(name, 0, column, bold=True)
        for row, name in enumerate(names, start=1):
            make_cell(name, row, 0, bold=True)
            for column in range(size):
                make_cell(str(int(rows[row - 1][column])), row, column + 1)

        for column in range(size + 1):
            cm_container.columnconfigure(column, weight=1)

    clear_confusion_matrix()

    def update_ui(*args):
        listbox.delete(0, tk.END)
        is_newest = sort_var.get() == "Mais novo"
        for res in sorted(METHOD_RESULTS, key=lambda x: x.timestamp, reverse=is_newest):
            listbox.insert(tk.END, res.get_display_name())

    def on_select(event):
        if not listbox.curselection():
            return
        idx = listbox.curselection()[0]
        is_newest = sort_var.get() == "Mais novo"
        selected = sorted(METHOD_RESULTS, key=lambda x: x.timestamp, reverse=is_newest)[
            idx
        ]

        metadata["method"].config(text=selected.method_name)
        metadata["timestamp"].config(
            text=selected.timestamp.strftime("%d/%m/%Y %H:%M:%S")
        )
        metadata["dataset"].config(text=selected.dataset_name or "-")
        metadata["samples"].config(
            text="-" if selected.n_samples is None else str(selected.n_samples)
        )
        metadata["features"].config(
            text="-" if selected.n_features is None else str(selected.n_features)
        )
        metadata["classes"].config(
            text="-" if selected.n_classes is None else str(selected.n_classes)
        )
        metadata["split"].config(
            text=(
                f"{selected.test_split:g}%" if selected.test_split is not None else "-"
            )
        )
        metadata["train_test"].config(
            text=(
                f"{selected.n_train} / {selected.n_test}"
                if selected.n_train is not None and selected.n_test is not None
                else "-"
            )
        )
        metadata["load_time"].config(text=format_duration(selected.load_time))
        metadata["train_time"].config(text=format_duration(selected.train_time))
        metadata["test_time"].config(text=format_duration(selected.test_time))
        metadata["total_time"].config(text=format_duration(selected.total_time))
        metadata["random_state"].config(
            text="-" if selected.random_state is None else str(selected.random_state)
        )
        metadata["preprocessing"].config(text=selected.preprocessing or "-")
        params_value.config(
            text=", ".join(
                f"{key}={value}" for key, value in selected.model_params.items()
            )
            or "-"
        )
        right_pane.config(text=f" 📈 {selected.method_name} ")

        if isinstance(selected.metrics, ClassificationMetrics):
            m = selected.metrics
            metric_labels["acc"].config(text=f"Acurácia: {m.accuracy*100:.2f}%")
            metric_labels["prec"].config(text=f"Precisão: {m.precision*100:.2f}%")
            metric_labels["rec"].config(text=f"Recall: {m.recall*100:.2f}%")
            metric_labels["f1"].config(text=f"F1-measure: {m.f1*100:.2f}%")
            render_confusion_matrix(m.confusion_matrix, selected.class_names)
        else:
            for key, name in metric_names.items():
                metric_labels[key].config(text=f"{name}: -")
            clear_confusion_matrix()

    listbox.bind("<<ListboxSelect>>", on_select)
    sort_var.trace_add("write", update_ui)
    parent.bind("<<UpdateResults>>", update_ui)


def main():
    root = tk.Tk()
    root.title("📊 Classificação e Regressão")
    root.geometry("860x760")
    root.minsize(700, 620)

    style = ttk.Style(root)
    style.theme_use("clam")

    top_menu(root)
    results_menu(root)

    root.mainloop()


if __name__ == "__main__":
    main()
