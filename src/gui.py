from tkinter import *
from tkinter import ttk


def side_window(frm: Frame):
    ttk.Label(frm, text="Carregar Arquivo").grid(column=0, row=0, sticky=W)
    ttk.Button(frm, text="Pesquisar").grid(column=1, row=0, sticky=W)

    options = ["Random Forest", "Multi Layer Perceptron"]
    ttk.Label(frm, text="Método", border=1).grid(column=0, row=1, sticky=W)
    method_box = ttk.Combobox(frm, values=options, state="readonly")
    method_box.grid(column=1, row=1, sticky=W)

    ttk.Label(frm, text="% Teste").grid(column=0, row=2, sticky=W)
    percentage_testing_box = ttk.Spinbox(frm, from_=0, to=100, increment=1)
    percentage_testing_box.grid(column=1, row=2, sticky=W)
    percentage_testing_box.set(20)

    # TODO: botão de executar

    # TODO: botão executar todos


def main():
    root = Tk()
    root.title("Classificação e Regressão")
    root.geometry("640x480")
    root.minsize(320, 240)

    frm = ttk.Frame(root, padding=10)
    frm.grid()
    frm.columnconfigure(1, weight=1, pad=10)
    frm.rowconfigure(0, weight=1, pad=10)

    side_window(frm)

    root.mainloop()
