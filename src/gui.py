from tkinter import *
from tkinter import ttk, filedialog
from src.enums import Methods

def open_file(path_entry):
    filename = filedialog.askopenfilename()
    if filename:
        path_entry.delete(0, END)
        path_entry.insert(0, filename)


def side_window(frm: Frame):
    common_kwargs = {"sticky": W, "padx": 5, "pady": 5}

    path_entry = ttk.Entry(frm, width=25)
    path_entry.grid(column=0, row=0, **common_kwargs)

    ttk.Button(frm, text="Pesquisar", command=lambda: open_file(path_entry)).grid(
      column=1, row=0, **common_kwargs
    )

    options = Methods.values()
    ttk.Label(frm, text="Método", border=1).grid(column=0, row=1, **common_kwargs)
    method_box = ttk.Combobox(frm, values=options, state="readonly")
    method_box.grid(column=1, row=1, **common_kwargs)

    ttk.Label(frm, text="% Teste").grid(column=0, row=2, **common_kwargs)
    percentage_testing_box = ttk.Spinbox(frm, from_=0, to=100, increment=1)
    percentage_testing_box.grid(column=1, row=2, **common_kwargs)
    percentage_testing_box.set(20)

    action_frame = ttk.Frame(frm)
    action_frame.grid(column=0, row=3, columnspan=2, sticky="sew", pady=(20, 0), padx=5)

    btn_executar = ttk.Button(action_frame, text="Executar Selecionado")
    btn_executar.pack(side=TOP, fill=X, pady=2)

    btn_executar_todos = ttk.Button(action_frame, text="Executar Todos")
    btn_executar_todos.pack(side=TOP, fill=X, pady=2)
  
def main():
    root = Tk()
    root.title("Classificação e Regressão")
    root.geometry("640x480")
    root.minsize(320, 240)

    frm = ttk.Frame(root, padding=10)
    frm.grid()
    frm.columnconfigure(1, weight=1, pad=10)
    frm.rowconfigure(0, weight=1, pad=10)
    frm.grid_configure(sticky=SW)

    side_window(frm)

    root.mainloop()
