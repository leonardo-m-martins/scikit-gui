"""Styling helpers and shared ttk style names for the GUI."""

from tkinter import ttk

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
