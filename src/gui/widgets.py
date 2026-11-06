"""Reusable Tk widgets."""

import tkinter as tk
from tkinter import ttk


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
