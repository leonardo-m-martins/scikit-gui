"""Results list and evaluation results view."""

import tkinter as tk
from tkinter import ttk

from src.gui.result_details import DetailsPanel, MetricsPanel
from src.gui.widgets import ScrollableFrame

SORT_NEWEST = "Mais novo"
SORT_OLDEST = "Mais velho"
SORT_OPTIONS = (SORT_NEWEST, SORT_OLDEST)


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
