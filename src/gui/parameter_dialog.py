"""Parameter editing dialog and its helper functions."""

import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.enums import Methods, Modes
from src.gui.widgets import ScrollableFrame
from src.ml.common import ModelSpec
from src.parameters import (
    CONFIG_DIR,
    ParameterFileError,
    build_payload,
    default_filename,
    parse_payload,
    write_payload,
)

try:
    from sklearn.utils._param_validation import Hidden, StrOptions
except ImportError:
    Hidden = None
    StrOptions = None


def string_parameter_options(
    model,
    parameter_name,
) -> tuple[list[str], bool] | None:
    constraints = getattr(model, "_parameter_constraints", {}).get(
        parameter_name,
        (),
    )

    options = []
    allows_none = False
    has_other_types = False

    for constraint in constraints:
        if StrOptions is not None and isinstance(constraint, StrOptions):
            options.extend(constraint.options)
        elif constraint is None or constraint is type(None):
            allows_none = True
        elif Hidden is not None and isinstance(constraint, Hidden):
            continue
        else:
            has_other_types = True

    if not options:
        return None

    result = sorted(set(options))
    if allows_none:
        result.append("None")

    return result, not has_other_types


class ParameterDialog(tk.Toplevel):
    def __init__(
        self,
        parent,
        spec: ModelSpec,
        mode: Modes,
        method: Methods,
        title: str,
    ):
        super().__init__(parent)
        self.spec = spec
        self.mode = mode
        self.method = method
        self.pending_preprocessing = spec.preprocessing
        self.result: ModelSpec | None = None
        self.vars: dict[str, tk.StringVar] = {}
        self.combobox_keys: set[str] = set()

        self.title(title)
        self.transient(parent)
        self.geometry("560x640")
        self.minsize(420, 300)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build_widgets()
        self.grab_set()
        self.focus_set()

    def _build_widgets(self):
        scroll = ScrollableFrame(self)
        scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        content = scroll.content
        content.columnconfigure(1, weight=1)

        model = self.spec.model_type()
        defaults = model.get_params()
        values = {**defaults, **self.spec.params}

        for row, key in enumerate(sorted(values)):
            ttk.Label(content, text=key).grid(
                row=row,
                column=0,
                sticky=tk.W,
                padx=(0, 10),
                pady=3,
            )

            value = values[key]
            option_info = string_parameter_options(model, key)

            if option_info is not None:
                options, read_only = option_info
                use_combobox = (value is None or isinstance(value, str)) and (
                    value is None or value in options
                )
            else:
                options = []
                read_only = False
                use_combobox = False

            if use_combobox:
                self.combobox_keys.add(key)
                var = tk.StringVar(
                    value="None" if value is None else value,
                )
                widget = ttk.Combobox(
                    content,
                    textvariable=var,
                    values=options,
                    state="readonly" if read_only else "normal",
                )
            else:
                var = tk.StringVar(value=repr(value))
                widget = ttk.Entry(content, textvariable=var)

            self.vars[key] = var
            widget.grid(
                row=row,
                column=1,
                sticky=tk.EW,
                pady=3,
            )

        action_frame = ttk.Frame(self)
        action_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        ttk.Button(
            action_frame,
            text="📥 Importar",
            command=self._import_params,
        ).pack(side=tk.LEFT)

        ttk.Button(
            action_frame,
            text="📤 Exportar",
            command=self._export_params,
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(
            action_frame,
            text="Cancelar",
            command=self._cancel,
        ).pack(side=tk.RIGHT)

        ttk.Button(
            action_frame,
            text="Aplicar",
            command=self._apply,
        ).pack(side=tk.RIGHT, padx=(0, 5))

    def _collect_params(self):
        params = {}

        for key, var in self.vars.items():
            text = var.get().strip()
            try:
                params[key] = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                params[key] = text

        try:
            self.spec.model_type(**params)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Invalid parameters",
                str(exc),
                parent=self,
            )
            return None

        return params

    def _apply(self):
        params = self._collect_params()
        if params is None:
            return

        self.result = self.spec.update_params(
            params,
            preprocessing=self.pending_preprocessing,
        )
        self.destroy()

    def _export_params(self):
        params = self._collect_params()
        if params is None:
            return

        payload = build_payload(
            self.mode,
            self.method,
            ModelSpec(
                model_type=self.spec.model_type,
                params=params,
                preprocessing=self.pending_preprocessing,
            ),
        )
        path = CONFIG_DIR / default_filename(self.mode, self.method)

        try:
            write_payload(path, payload)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror(
                "Exportar parâmetros",
                str(exc),
                parent=self,
            )
            return

        messagebox.showinfo(
            "Exportar parâmetros",
            f"Parâmetros de {self.method.value} salvos em:\n{path}",
            parent=self,
        )

    def _import_params(self):
        filename = filedialog.askopenfilename(
            parent=self,
            filetypes=[
                ("Arquivo JSON", "*.json"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not filename:
            return

        try:
            with open(filename, encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Importar parâmetros",
                str(exc),
                parent=self,
            )
            return

        file_method = payload.get("method") if isinstance(payload, dict) else None
        if file_method != self.method.value:
            messagebox.showerror(
                "Importar parâmetros",
                f"O arquivo é do método {file_method!r}, "
                f"mas este diálogo edita {self.method.value!r}.",
                parent=self,
            )
            return

        try:
            _, imported = parse_payload(
                payload,
                self.mode,
                {self.method: self.spec},
            )
        except ParameterFileError as exc:
            messagebox.showerror(
                "Importar parâmetros",
                str(exc),
                parent=self,
            )
            return

        self.pending_preprocessing = imported.preprocessing
        defaults = self.spec.model_type().get_params()
        values = {**defaults, **imported.params}

        for key, var in self.vars.items():
            value = values[key]
            if key in self.combobox_keys:
                var.set("None" if value is None else str(value))
            else:
                var.set(repr(value))

        messagebox.showinfo(
            "Importar parâmetros",
            f"Parâmetros de {self.method.value} carregados.\n"
            "Clique em Aplicar para confirmar.\n"
            f"Pré-processamento: {imported.preprocessing.value}",
            parent=self,
        )

    def _cancel(self):
        self.result = None
        self.destroy()
