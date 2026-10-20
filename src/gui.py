import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Any
from src.enums import Methods
from src.ml_clas_sk import run_individual_method, run_all_methods
from src.shared import MethodResult, Metrics, ClassificationMetrics

DEFAULT_FILE = "./data/spambase.data"
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

    labels = {
        "acc": ttk.Label(right_pane, text="Acurácia: -", font=("", 10, "bold")),
        "prec": ttk.Label(right_pane, text="Precisão: -", font=("", 10, "bold")),
        "rec": ttk.Label(right_pane, text="Recall: -", font=("", 10, "bold")),
        "f1": ttk.Label(right_pane, text="F1-measure: -", font=("", 10, "bold")),
        "cm": ttk.Label(right_pane, text="Matriz de Confusão:\n-", justify=tk.LEFT),
    }

    for lbl in labels.values():
        lbl.pack(anchor=tk.W, pady=4)

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

        if isinstance(selected.metrics, ClassificationMetrics):
            m = selected.metrics
            labels["acc"].config(text=f"Acurácia: {m.accuracy*100:.2f}%")
            labels["prec"].config(text=f"Precisão: {m.precision*100:.2f}%")
            labels["rec"].config(text=f"Recall: {m.recall*100:.2f}%")
            labels["f1"].config(text=f"F1-measure: {m.f1*100:.2f}%")
            labels["cm"].config(text=f"Matriz de Confusão:\n{m.confusion_matrix}")

    listbox.bind("<<ListboxSelect>>", on_select)
    sort_var.trace_add("write", update_ui)
    parent.bind("<<UpdateResults>>", update_ui)


def main():
    root = tk.Tk()
    root.title("📊 Classificação e Regressão")
    root.geometry("680x580")
    root.minsize(550, 480)

    style = ttk.Style(root)
    style.theme_use("clam")

    top_menu(root)
    results_menu(root)

    root.mainloop()


if __name__ == "__main__":
    main()
