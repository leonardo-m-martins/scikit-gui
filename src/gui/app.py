"""Application entry point and main window orchestration."""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from src.enums import Methods, Modes
from src.gui.config_panel import ConfigPanel
from src.gui.data_summary import DataSummaryView
from src.gui.data_table import DataView
from src.gui.model_configs import MODE_CONFIGS
from src.gui.results import ResultsView
from src.gui.styles import configure_styles
from src.ml.common import ModeConfig, run_all_methods, run_method
from src.parameters import load_method_spec, load_mode_specs
from src.shared import MethodResult


class App:
    def __init__(self, root):
        self.root = root
        self.results: list[MethodResult] = []
        self._busy = False
        self._worker_queue = queue.Queue()
        configure_styles(root)

        self.config = ConfigPanel(
            root,
            on_run_selected=self.run_selected,
            on_run_all=self.run_all,
            on_dataset_loaded=self._on_dataset_loaded,
        )
        self.config.pack(fill=tk.X, padx=20, pady=(20, 10))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.summary_view = DataSummaryView(self.notebook)
        self.data_view = DataView(self.notebook)
        self.results_view = ResultsView(self.notebook)
        self.notebook.add(self.summary_view, text=" 📄 Resumo Dados ")
        self.notebook.add(self.data_view, text=" 📋 Dados ")
        self.notebook.add(self.results_view, text=" 📈 Resultados ")

        self.config.reload_dataset()

    def _on_dataset_loaded(self, dataset, mode):
        self.summary_view.render(dataset, mode)
        self.data_view.render(dataset)

    def run_selected(self):
        dataset, split, mode, method, use_default = self.config.get_values()
        mode_config = MODE_CONFIGS[mode]

        if not method:
            messagebox.showwarning("Method", "Select a method first.")
            return

        if not use_default:
            try:
                spec = load_method_spec(
                    mode,
                    Methods(method),
                    mode_config.specs,
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("Configuração", str(exc))
                return

            mode_config = ModeConfig(
                {Methods(method): spec},
                mode_config.metrics_fn,
            )

        def work():
            return run_method(
                method,
                dataset,
                split,
                mode_config,
                stratify=mode is Modes.CLASSIFICATION,
            )

        self._run_in_background(
            work,
            self._append_result,
            total=1,
            method_name=method,
        )

    def run_all(self):
        dataset, split, mode, _, use_default = self.config.get_values()
        mode_config = MODE_CONFIGS[mode]

        if not use_default:
            try:
                specs = load_mode_specs(mode, mode_config.specs)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Configuração", str(exc))
                return

            mode_config = ModeConfig(specs, mode_config.metrics_fn)

        methods = [method.value for method in mode_config.specs]

        def work():
            return run_all_methods(
                dataset,
                split,
                mode_config,
                stratify=mode is Modes.CLASSIFICATION,
                on_method_start=lambda name: self._worker_queue.put(
                    ("method_started", name)
                ),
                on_method_complete=lambda: self._worker_queue.put(
                    ("method_completed", None)
                ),
            )

        self._run_in_background(
            work,
            self._append_results,
            total=len(methods),
            method_name=methods[0] if methods else "",
        )

    def _run_in_background(
        self,
        work,
        on_success,
        *,
        total: int = 1,
        method_name: str = "",
    ):
        if self._busy:
            return

        self._clear_worker_queue()
        self._busy = True
        self.config.set_running(True, total=total, method_name=method_name)

        def worker():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                self._worker_queue.put(("error", exc))
            else:
                self._worker_queue.put(("success", result))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_worker, on_success)

    def _poll_worker(self, on_success):
        try:
            kind, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker, on_success)
            return

        if kind == "method_started":
            self.config.set_current_method(payload)
            self.root.after(100, self._poll_worker, on_success)
            return

        if kind == "method_completed":
            self.config.mark_method_completed()
            self.root.after(100, self._poll_worker, on_success)
            return

        self._busy = False
        self.config.set_running(False)

        if kind == "error":
            messagebox.showerror("Execution error", str(payload))
            return

        on_success(payload)

    def _clear_worker_queue(self):
        while True:
            try:
                self._worker_queue.get_nowait()
            except queue.Empty:
                return

    def _append_result(self, result):
        if result is None:
            return
        self.results.append(result)
        self.results_view.refresh(self.results)
        self.results_view.select_result(result)
        self.notebook.select(self.results_view)

    def _append_results(self, results):
        self.results.extend(results)
        self.results_view.refresh(self.results)
        if results:
            self.results_view.select_result(results[-1])
            self.notebook.select(self.results_view)


def maximize_window(root):
    root.update_idletasks()
    system = root.tk.call("tk", "windowingsystem")

    try:
        if system == "x11":
            root.attributes("-zoomed", True)
        else:
            root.state("zoomed")
    except tk.TclError:
        try:
            if system == "x11":
                root.state("zoomed")
            else:
                root.attributes("-zoomed", True)
        except tk.TclError:
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.geometry(f"{screen_width}x{screen_height}+0+0")


def main():
    root = tk.Tk()
    root.title("📊 Classificação e Regressão")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    width = min(1000, max(640, screen_width - 80))
    height = min(800, max(480, screen_height - 80))
    root.geometry(f"{width}x{height}+0+0")
    root.minsize(min(760, width), min(560, height))

    App(root)
    maximize_window(root)

    root.mainloop()
