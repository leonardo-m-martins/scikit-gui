import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import gui
from src.shared import ClassificationMetrics, MethodResult


@pytest.fixture(autouse=True)
def restore_method_results():
    saved = list(gui.METHOD_RESULTS)
    gui.METHOD_RESULTS.clear()
    yield
    gui.METHOD_RESULTS[:] = saved


@pytest.fixture
def arff_factory(tmp_path):
    def make(rows, labels=("yes", "no"), numeric_class=False, name="tiny"):
        class_declaration = "numeric" if numeric_class else "{" + ",".join(labels) + "}"
        lines = [
            f"@relation {name}",
            "",
            "@attribute f1 numeric",
            "@attribute f2 numeric",
            f"@attribute class {class_declaration}",
            "",
            "@data",
        ]
        lines.extend(",".join(str(value) for value in row) for row in rows)
        path = tmp_path / f"{name}.arff"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return make


@pytest.fixture
def tiny_arff(arff_factory):
    rows = []
    for index in range(20):
        rows.append((index / 10, (index % 3) / 10, "yes"))
        rows.append((-index / 10, ((index + 1) % 3) / 10, "no"))
    return arff_factory(rows)


@pytest.fixture
def result_factory():
    def make(method_name="PART", *, metrics=None, class_names=None, **overrides):
        names = list(class_names) if class_names is not None else ["yes", "no"]
        if metrics is None:
            metrics = ClassificationMetrics(
                accuracy=0.9,
                precision=0.8,
                recall=0.7,
                f1=0.75,
                confusion_matrix=np.array([[8, 2], [1, 9]]),
            )
        n_test = int(np.asarray(metrics.confusion_matrix).sum())
        values = {
            "dataset_name": "tiny.arff",
            "n_samples": 40,
            "n_features": 2,
            "n_classes": len(names),
            "n_train": 40 - n_test,
            "n_test": n_test,
            "test_split": 50,
            "load_time": 0.0123,
            "train_time": 0.5,
            "test_time": 0.002,
            "random_state": 42,
            "model_params": {"criterion": "entropy"},
            "preprocessing": None,
            "class_names": names,
        }
        values.update(overrides)
        return MethodResult(method_name, metrics, **values)

    return make


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    root._callback_errors = []
    root.report_callback_exception = lambda exc, value, traceback: (
        root._callback_errors.append((exc, value, traceback))
    )
    yield root
    root.destroy()


@pytest.fixture
def results_app(tk_root):
    gui.top_menu(tk_root)
    gui.results_menu(tk_root)
    tk_root.update()
    return tk_root
