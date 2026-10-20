import sys
import tkinter as tk
from tkinter import ttk, filedialog
from src.enums import Methods
from src.ml_clas_sk import run_individual_method, run_all_methods

DEFAULT_FILE = "./data/spambase.data"


class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass


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

    ttk.Button(
        action_frame,
        text="▶️ Executar Selecionado",
        command=lambda: run_individual_method(
            path_entry.get(), int(percentage_box.get()), method_box.get()
        ),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    ttk.Button(
        action_frame,
        text="🚀 Executar Todos",
        command=lambda: run_all_methods(path_entry.get(), int(percentage_box.get())),
    ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    # Redirects terminal output automatically to the metrics UI text widget
    metrics_frm = ttk.LabelFrame(parent, text=" 📈 Métricas de Avaliação ", padding=10)
    metrics_frm.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    metrics_text = tk.Text(
        metrics_frm, wrap=tk.WORD, state=tk.DISABLED, height=12, bg="#f8f9fa"
    )
    scrollbar = ttk.Scrollbar(metrics_frm, command=metrics_text.yview)
    metrics_text.configure(yscrollcommand=scrollbar.set)

    metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    sys.stdout = OutputRedirector(metrics_text)


def main():
    root = tk.Tk()
    root.title("📊 Classificação e Regressão")
    root.geometry("680x580")
    root.minsize(550, 480)

    style = ttk.Style(root)
    style.theme_use("clam")

    top_menu(root)
    root.mainloop()


if __name__ == "__main__":
    main()
