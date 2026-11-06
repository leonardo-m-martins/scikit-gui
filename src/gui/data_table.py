"""Dataset table view with pagination."""

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from src.gui.styles import STYLE_SECTION

PAGE_SIZE_OPTIONS = (50, 100, 200, 500)
DEFAULT_PAGE_SIZE = 100


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
