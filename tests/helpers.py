import tkinter as tk
from collections.abc import Iterator


def walk(widget: tk.Misc) -> Iterator[tk.Misc]:
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def find_widgets(widget: tk.Misc, widget_class: str) -> list[tk.Misc]:
    return [item for item in walk(widget) if item.winfo_class() == widget_class]


def find_widget(widget: tk.Misc, widget_class: str) -> tk.Misc:
    matches = find_widgets(widget, widget_class)
    assert matches, f"no {widget_class} found"
    return matches[0]


def label_texts(widget: tk.Misc) -> list[str]:
    return [item.cget("text") for item in find_widgets(widget, "TLabel")]


def find_label(widget: tk.Misc, text: str) -> tk.Misc:
    matches = [
        item for item in find_widgets(widget, "TLabel") if text in item.cget("text")
    ]
    assert matches, f"label containing {text!r} not found"
    return matches[0]


def find_button(widget: tk.Misc, text: str) -> tk.Misc:
    matches = [
        item for item in find_widgets(widget, "TButton") if text in item.cget("text")
    ]
    assert matches, f"button containing {text!r} not found"
    return matches[0]


def refresh_results(root: tk.Misc) -> None:
    root.app.results_view.refresh(root.app.results)
    root.update()


def select_result(root: tk.Misc, *, index: int | None = None, text: str | None = None):
    listbox = find_widget(root, "Listbox")
    entries = listbox.get(0, tk.END)
    if text is not None:
        index = next(i for i, entry in enumerate(entries) if text in entry)
    if index is None:
        index = 0
    listbox.selection_clear(0, tk.END)
    listbox.selection_set(index)
    listbox.event_generate("<<ListboxSelect>>")
    root.update()


def confusion_cells(root: tk.Misc) -> list[str]:
    title = find_label(root, "Matriz de Confusão:")
    container = next(
        child
        for child in title.master.winfo_children()
        if child.winfo_class() == "TFrame"
    )
    cells = [
        child for child in container.winfo_children() if child.winfo_class() == "TLabel"
    ]
    cells.sort(
        key=lambda cell: (
            int(cell.grid_info()["row"]),
            int(cell.grid_info()["column"]),
        )
    )
    return [str(cell.cget("text")) for cell in cells]
