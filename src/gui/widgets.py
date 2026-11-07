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


class ModeSlider(ttk.Frame):
    """Binary slider that chooses between two configuration sources."""

    def __init__(
        self,
        parent,
        left_text: str,
        right_text: str,
        variable: tk.BooleanVar | None = None,
        command=None,
    ):
        super().__init__(parent)
        self.variable = variable if variable is not None else tk.BooleanVar(False)
        self._command = command
        self._updating = False

        self.left_label = ttk.Label(self, text=left_text)
        self.left_label.pack(side=tk.LEFT, padx=(0, 6))
        self.left_label.bind("<Button-1>", lambda _event: self.set(True))

        self.scale_var = tk.DoubleVar(value=0 if self.variable.get() else 1)
        self.scale = ttk.Scale(
            self,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            length=90,
            variable=self.scale_var,
            command=self._on_scale,
        )
        self.scale.pack(side=tk.LEFT)

        self.right_label = ttk.Label(self, text=right_text)
        self.right_label.pack(side=tk.LEFT, padx=(6, 0))
        self.right_label.bind("<Button-1>", lambda _event: self.set(False))

        self.variable.trace_add("write", self._on_variable_changed)
        self._update_labels()

    def _on_scale(self, value):
        if self._updating:
            return

        target = 0.0 if float(value) < 0.5 else 1.0
        self._updating = True
        self.scale_var.set(target)
        self.variable.set(target == 0.0)
        self._updating = False
        self._update_labels()
        self._notify()

    def _on_variable_changed(self, *_args):
        if self._updating:
            return

        self._updating = True
        self.scale_var.set(0 if self.variable.get() else 1)
        self._updating = False
        self._update_labels()
        self._notify()

    def _update_labels(self):
        active_font = ("", 9, "bold")
        inactive_font = ("", 9)
        is_default = self.variable.get()
        self.left_label.configure(
            font=active_font if is_default else inactive_font,
        )
        self.right_label.configure(
            font=inactive_font if is_default else active_font,
        )

    def _notify(self):
        if self._command is not None:
            self._command()

    def get(self) -> bool:
        return self.variable.get()

    def set(self, use_default: bool):
        self._updating = True
        self.variable.set(use_default)
        self.scale_var.set(0 if use_default else 1)
        self._updating = False
        self._update_labels()
        self._notify()

    def set_enabled(self, enabled: bool):
        self.scale.state(["!disabled"] if enabled else ["disabled"])
        for label in (self.left_label, self.right_label):
            label.configure(foreground="" if enabled else "gray")
