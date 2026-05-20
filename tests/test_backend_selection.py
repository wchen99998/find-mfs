import importlib

import pytest

from find_mfs import FormulaFinder


def test_formula_finder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        FormulaFinder("CHNOPS", backend="unknown")

    finder = FormulaFinder("CHNOPS")
    with pytest.raises(ValueError, match="Unknown backend"):
        finder.find_formulae(180.063, error_ppm=5.0, backend="unknown")


def test_python_backend_is_default_and_does_not_import_rust(monkeypatch):
    def fail_if_rust_imported(name, *args, **kwargs):
        if name == "find_mfs._rust":
            raise AssertionError("python backend should not import find_mfs._rust")
        return original_import_module(name, *args, **kwargs)

    original_import_module = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", fail_if_rust_imported)

    results = FormulaFinder("CHNOPS").find_formulae(
        180.063,
        error_ppm=5.0,
        max_results=5,
    )
    assert len(results) == 5


def test_rust_backend_reports_clear_error_when_extension_missing(monkeypatch):
    def import_without_rust(name, *args, **kwargs):
        if name == "find_mfs._rust":
            raise ImportError("simulated missing rust extension")
        return original_import_module(name, *args, **kwargs)

    original_import_module = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", import_without_rust)

    with pytest.raises(ImportError, match="Rust backend requested"):
        FormulaFinder("CHNOPS", backend="rust").find_formulae(
            180.063,
            error_ppm=5.0,
            max_results=5,
        )
