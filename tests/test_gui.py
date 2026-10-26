from datetime import datetime

import numpy as np

from src import gui
from src.enums import Methods
from src.shared import ClassificationMetrics, MethodResult, Metrics
from tests.helpers import (
    confusion_cells,
    find_button,
    find_label,
    find_widget,
    find_widgets,
    label_texts,
    refresh_results,
    select_result,
)


def test_results_app_constructs(results_app):
    assert find_widget(results_app, "Listbox")
    assert find_label(results_app, "Matriz de Confusão:")
    assert find_button(results_app, "Executar Selecionado")
    assert find_button(results_app, "Executar Todos")


def test_selection_populates_details_and_confusion_table(results_app, result_factory):
    gui.METHOD_RESULTS.append(result_factory())
    refresh_results(results_app)
    select_result(results_app, index=0)

    texts = label_texts(results_app)
    for expected in [
        "PART",
        "tiny.arff",
        "40",
        "2",
        "20 / 20",
        "50%",
        "12.30 ms",
        "0.50 s",
        "2.00 ms",
        "0.51 s",
        "42",
        "criterion=entropy",
        "Acurácia: 90.00%",
        "Precisão: 80.00%",
        "Recall: 70.00%",
        "F1-measure: 75.00%",
    ]:
        assert expected in texts

    assert confusion_cells(results_app) == [
        "Real \\ Previsto",
        "yes",
        "no",
        "yes",
        "8",
        "2",
        "no",
        "1",
        "9",
    ]


def test_non_classification_result_clears_metrics(results_app, result_factory):
    gui.METHOD_RESULTS.append(result_factory(method_name="Com métricas"))
    refresh_results(results_app)
    select_result(results_app, text="Com métricas")
    assert confusion_cells(results_app)[1:] == [
        "yes",
        "no",
        "yes",
        "8",
        "2",
        "no",
        "1",
        "9",
    ]

    gui.METHOD_RESULTS.append(MethodResult("Sem métricas", Metrics()))
    refresh_results(results_app)
    select_result(results_app, text="Sem métricas")

    texts = label_texts(results_app)
    assert "Acurácia: -" in texts
    assert "Precisão: -" in texts
    assert "Recall: -" in texts
    assert "F1-measure: -" in texts
    assert confusion_cells(results_app) == ["-"]


def test_multiclass_confusion_table(results_app, result_factory):
    metrics = ClassificationMetrics(
        0.9,
        0.8,
        0.7,
        0.75,
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
    )
    gui.METHOD_RESULTS.append(
        result_factory(
            method_name="Multiclasse",
            metrics=metrics,
            class_names=["a", "b", "c"],
        )
    )
    refresh_results(results_app)
    select_result(results_app, index=0)

    assert confusion_cells(results_app) == [
        "Real \\ Previsto",
        "a",
        "b",
        "c",
        "a",
        "1",
        "0",
        "0",
        "b",
        "0",
        "2",
        "0",
        "c",
        "0",
        "0",
        "3",
    ]


def test_confusion_table_falls_back_to_generated_class_names(
    results_app, result_factory
):
    gui.METHOD_RESULTS.append(
        result_factory(method_name="Sem nomes", class_names=["only-one"])
    )
    refresh_results(results_app)
    select_result(results_app, index=0)

    assert confusion_cells(results_app)[1:3] == ["Classe 0", "Classe 1"]


def test_sorting_changes_listbox_order(results_app, result_factory):
    older = result_factory(method_name="Modelo Antigo")
    older.timestamp = datetime(2026, 1, 1, 12, 0, 0)
    newer = result_factory(method_name="Modelo Novo")
    newer.timestamp = datetime(2026, 1, 2, 12, 0, 0)
    gui.METHOD_RESULTS.extend([older, newer])
    refresh_results(results_app)

    listbox = find_widget(results_app, "Listbox")
    assert "Modelo Novo" in listbox.get(0)
    assert "Modelo Antigo" in listbox.get(1)

    sort_box = next(
        widget
        for widget in find_widgets(results_app, "TCombobox")
        if "Mais novo" in widget.cget("values")
    )
    sort_box.set("Mais velho")
    results_app.update()

    assert "Modelo Antigo" in listbox.get(0)
    assert "Modelo Novo" in listbox.get(1)


def test_open_file_button_updates_entry(results_app, monkeypatch, tmp_path):
    chosen = str(tmp_path / "chosen.arff")
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda: chosen)

    find_button(results_app, "Procurar").invoke()
    results_app.update()

    assert find_widget(results_app, "TEntry").get() == chosen


def test_open_file_cancel_keeps_entry(results_app, monkeypatch):
    entry = find_widget(results_app, "TEntry")
    previous = entry.get()
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda: "")

    find_button(results_app, "Procurar").invoke()
    results_app.update()

    assert entry.get() == previous


def test_execute_selected_passes_ui_values(results_app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui, "run_individual_method", lambda *args: calls.append(args))
    path = str(tmp_path / "data.arff")
    entry = find_widget(results_app, "TEntry")
    entry.delete(0, "end")
    entry.insert(0, path)
    method_box = next(
        widget
        for widget in find_widgets(results_app, "TCombobox")
        if Methods.DECISION_TREE.value in widget.cget("values")
    )
    method_box.set(Methods.DECISION_TREE.value)
    find_widget(results_app, "TSpinbox").set(30)

    find_button(results_app, "Executar Selecionado").invoke()

    assert len(calls) == 1
    called_path, split, method, results = calls[0]
    assert called_path == path
    assert split == 30
    assert method == Methods.DECISION_TREE.value
    assert results is gui.METHOD_RESULTS


def test_execute_all_passes_ui_values(results_app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui, "run_all_methods", lambda *args: calls.append(args))
    path = str(tmp_path / "data.arff")
    entry = find_widget(results_app, "TEntry")
    entry.delete(0, "end")
    entry.insert(0, path)
    find_widget(results_app, "TSpinbox").set(40)

    find_button(results_app, "Executar Todos").invoke()

    assert len(calls) == 1
    called_path, split, results = calls[0]
    assert called_path == path
    assert split == 40
    assert results is gui.METHOD_RESULTS


def test_mousewheel_over_tcl_native_widget_is_ignored(results_app, result_factory):
    gui.METHOD_RESULTS.append(result_factory())
    refresh_results(results_app)
    select_result(results_app, index=0)
    results_app.deiconify()
    results_app.geometry("700x400")
    results_app.update()

    scrollbar = find_widget(results_app, "TScrollbar")
    assert scrollbar.winfo_ismapped()

    results_app.tk.call("frame", ".native_frame")
    results_app.tk.call(
        "place", ".native_frame", "-x", 0, "-y", 0, "-width", 100, "-height", 100
    )
    results_app.update()
    results_app.tk.call("event", "generate", ".native_frame", "<Button-4>")
    results_app.update()

    assert results_app._callback_errors == []
