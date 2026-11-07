"""Model and dataset configuration panel."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.enums import Methods, Modes, Preprocessors
from src.gui.model_configs import MODE_CONFIGS, method_values
from src.gui.parameter_dialog import ParameterDialog
from src.gui.widgets import ModeSlider
from src.loader import load_dataset
from src.parameters import config_path, load_method_spec, save_method_spec

DEFAULT_FILE = "./data/spambase.arff"


class ConfigPanel(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        on_run_selected,
        on_run_all,
        on_dataset_loaded=None,
    ):
        super().__init__(parent, text=" ⚙️ Configurações do Modelo ", padding=20)
        self._on_run_selected = on_run_selected
        self._on_run_all = on_run_all
        self._on_dataset_loaded = on_dataset_loaded

        self.path_var = tk.StringVar(value=DEFAULT_FILE)
        self.mode_var = tk.StringVar(value=Modes.CLASSIFICATION.value)
        self.method_var = tk.StringVar()
        self.preprocessing_var = tk.StringVar(value=Preprocessors.NONE.value)
        self.split_var = tk.IntVar(value=20)
        self.default_var = tk.BooleanVar(value=True)
        self.dataset = None
        self._loaded_mode = Modes.CLASSIFICATION
        self._running = False
        self._progress_total = 0
        self._progress_completed = 0
        self._current_method = ""
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="📁 Arquivo de Dados:").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10), padx=(0, 10)
        )
        self.path_entry = ttk.Entry(self, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 10), padx=(0, 10))
        self.browse_button = ttk.Button(
            self,
            text="🔍 Procurar",
            command=self._browse,
        )
        self.browse_button.grid(row=0, column=2, pady=(0, 10))

        ttk.Label(self, text="🎚️ Modo:").grid(
            row=1, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.mode_box = ttk.Combobox(
            self,
            textvariable=self.mode_var,
            values=Modes.values(),
            state="readonly",
        )
        self.mode_box.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(self, text="🛠️ Método:").grid(
            row=2, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.method_box = ttk.Combobox(
            self,
            textvariable=self.method_var,
            values=method_values(Modes.CLASSIFICATION),
            state="readonly",
        )
        self.method_box.grid(
            row=2,
            column=1,
            sticky=tk.EW,
            pady=10,
        )
        self.method_box.bind("<<ComboboxSelected>>", self._on_method_changed)

        self.params_button = ttk.Button(
            self,
            text="⚙️ Parâmetros",
            command=self._open_params,
        )
        self.params_button.grid(
            row=2,
            column=2,
            padx=(10, 0),
            pady=10,
        )

        ttk.Label(self, text="🗒️ Pré-processamento:").grid(
            row=3, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.preprocessing_box = ttk.Combobox(
            self,
            textvariable=self.preprocessing_var,
            values=Preprocessors.values(),
            state=tk.DISABLED,
        )
        self.preprocessing_box.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )
        self.preprocessing_box.bind(
            "<<ComboboxSelected>>",
            self._on_preprocessing_changed,
        )

        ttk.Label(self, text="📊 % de Teste:").grid(
            row=4, column=0, sticky=tk.W, pady=10, padx=(0, 10)
        )
        self.split_box = ttk.Spinbox(
            self,
            textvariable=self.split_var,
            from_=0,
            to=100,
            increment=1,
        )
        self.split_box.grid(
            row=4,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=10,
        )

        self.mode_slider = ModeSlider(
            self,
            "Padrão",
            "CONFIG.json",
            variable=self.default_var,
            command=self._on_config_mode_changed,
        )
        self.mode_slider.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        action_frame = ttk.Frame(self)
        action_frame.grid(row=6, column=0, columnspan=3, sticky=tk.EW, pady=(25, 0))

        self.run_selected_button = ttk.Button(
            action_frame,
            text="▶️ Executar Selecionado",
            command=self._on_run_selected,
        )
        self.run_selected_button.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 5),
        )

        self.run_all_button = ttk.Button(
            action_frame,
            text="🚀 Executar Todos",
            command=self._on_run_all,
        )
        self.run_all_button.pack(
            side=tk.RIGHT,
            fill=tk.X,
            expand=True,
            padx=(5, 0),
        )

        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(
            row=7,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(10, 0),
        )

        self.spinner = ttk.Progressbar(
            self.status_frame,
            mode="indeterminate",
            length=100,
        )
        self.spinner.pack(side=tk.LEFT, padx=(0, 8))

        self.current_method_var = tk.StringVar()
        ttk.Label(
            self.status_frame,
            textvariable=self.current_method_var,
        ).pack(side=tk.LEFT)

        self.progress_var = tk.StringVar()
        ttk.Label(
            self.status_frame,
            textvariable=self.progress_var,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.status_frame.grid_remove()

    def set_running(
        self,
        running: bool,
        total: int = 1,
        method_name: str = "",
    ):
        self._running = running

        if running:
            self._set_controls_enabled(False)
            self._progress_total = total
            self._progress_completed = 0
            self._current_method = method_name
            self.status_frame.grid()
            self.spinner.start(12)
            self._update_progress_label()
            return

        self.spinner.stop()
        self.status_frame.grid_remove()
        self.current_method_var.set("")
        self.progress_var.set("")
        self._set_controls_enabled(True)

    def set_current_method(self, method_name: str):
        self._current_method = method_name
        self._update_progress_label()

    def mark_method_completed(self):
        if not self._running:
            return

        self._progress_completed = min(
            self._progress_completed + 1,
            self._progress_total,
        )
        self._update_progress_label()

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        self.path_entry.configure(state=state)
        self.browse_button.configure(state=state)
        self.split_box.configure(state=state)
        self.run_selected_button.configure(state=state)
        self.run_all_button.configure(state=state)
        self.params_button.configure(state=state)
        self.mode_slider.set_enabled(enabled)

        combo_state = "readonly" if enabled else tk.DISABLED
        self.mode_box.configure(state=combo_state)
        self.method_box.configure(state=combo_state)

        preprocessing_state = (
            "readonly"
            if enabled and not self.default_var.get() and self.method_var.get()
            else tk.DISABLED
        )
        self.preprocessing_box.configure(state=preprocessing_state)

    def _update_progress_label(self):
        self.current_method_var.set(self._current_method)

        if not self._running or self._progress_total <= 1:
            self.progress_var.set("")
            return

        self.progress_var.set(f"{self._progress_completed}/{self._progress_total}")

    def get_values(self):
        return (
            self.dataset,
            int(self.split_var.get()),
            Modes(self.mode_var.get()),
            self.method_var.get(),
            self.default_var.get(),
        )

    def _open_params(self):
        method = self.method_var.get()
        if not method:
            messagebox.showwarning("Parameters", "Select a method first.")
            return

        mode = Modes(self.mode_var.get())
        method_enum = Methods(method)
        base_spec = MODE_CONFIGS[mode].specs.get(method_enum)
        if base_spec is None:
            messagebox.showwarning("Parameters", "Method is not supported.")
            return

        spec = base_spec
        if not self.default_var.get() and config_path(mode, method_enum).is_file():
            try:
                spec = load_method_spec(mode, method_enum, MODE_CONFIGS[mode].specs)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Parâmetros", str(exc))
                return

        dialog = ParameterDialog(
            self,
            spec,
            mode=mode,
            method=method_enum,
            title=f"Parâmetros — {method}",
        )
        self.wait_window(dialog)

        if dialog.result is not None:
            try:
                save_method_spec(mode, method_enum, dialog.result)
            except OSError as exc:
                messagebox.showerror("Parâmetros", str(exc))
                return

            self.mode_slider.set(False)
            self._update_preprocessing_display()

    def _browse(self):
        if filename := filedialog.askopenfilename(
            filetypes=[
                ("Arquivos ARFF", "*.arff"),
                ("Todos os arquivos", "*.*"),
            ],
        ):
            self.path_var.set(filename)
            self._update_dataset()

    def _on_mode_changed(self, _event=None):
        previous_mode = self._loaded_mode
        self._update_method_values()

        if not self._update_dataset():
            self.mode_var.set(previous_mode.value)
            self._update_method_values()

    def _update_method_values(self):
        mode = Modes(self.mode_var.get())
        self.method_box.configure(values=method_values(mode))
        self.method_var.set("")
        self._on_method_changed()

    def _on_method_changed(self, _event=None):
        self._update_preprocessing_display()

    def _on_config_mode_changed(self):
        self._update_preprocessing_display()

    def _effective_spec(self):
        method = self.method_var.get()
        if not method:
            return None

        mode = Modes(self.mode_var.get())
        method_enum = Methods(method)
        base_spec = MODE_CONFIGS[mode].specs.get(method_enum)
        if base_spec is None:
            return None

        if self.default_var.get():
            return base_spec

        try:
            return load_method_spec(mode, method_enum, MODE_CONFIGS[mode].specs)
        except (OSError, ValueError):
            return base_spec

    def _update_preprocessing_display(self):
        spec = self._effective_spec()
        if spec is None:
            self.preprocessing_var.set(Preprocessors.NONE.value)
            self.preprocessing_box.configure(state=tk.DISABLED)
            return

        self.preprocessing_var.set(spec.preprocessing.value)
        state = tk.DISABLED if self.default_var.get() else "readonly"
        self.preprocessing_box.configure(state=state)

    def _on_preprocessing_changed(self, _event=None):
        method = self.method_var.get()
        if not method or self.default_var.get():
            return

        mode = Modes(self.mode_var.get())
        method_enum = Methods(method)
        base_specs = MODE_CONFIGS[mode].specs
        try:
            spec = load_method_spec(mode, method_enum, base_specs)
        except (OSError, ValueError):
            spec = base_specs[method_enum]

        preprocessing = Preprocessors(self.preprocessing_var.get())
        spec = spec.update_params({}, preprocessing=preprocessing)

        try:
            save_method_spec(mode, method_enum, spec)
        except OSError as exc:
            messagebox.showerror("Pré-processamento", str(exc))

        self._update_preprocessing_display()

    def reload_dataset(self) -> bool:
        return self._update_dataset()

    def _update_dataset(self) -> bool:
        mode = Modes(self.mode_var.get())

        try:
            dataset = load_dataset(self.path_var.get(), mode=mode)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return False

        self.dataset = dataset
        self._loaded_mode = mode

        if self._on_dataset_loaded is not None:
            self._on_dataset_loaded(dataset, mode)

        return True
