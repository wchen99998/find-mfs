from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
import pytest
from molmass import Formula

import find_mfs
from find_mfs import FormulaFinder, FormulaPrior

pytest.importorskip("find_mfs._rust")


METABOLITE_CORPUS = [
    "C6H12O6",
    "C12H22O11",
    "C27H46O",
    "C5H9NO4",
    "C3H7NO2",
    "C6H13NO2",
    "C4H8N2O3",
    "C5H11NO2",
    "C9H11NO3",
    "C5H9NO2",
    "C10H16N5O13P3",
    "C21H28O5",
    "C16H18N2O4S",
    "C8H10N4O2",
    "C20H25N3O",
]


@lru_cache
def _finder(elements: str, backend: str) -> FormulaFinder:
    return FormulaFinder(elements, backend=backend)


def _counts_for_symbols(formula, symbols: list[str]) -> tuple[int, ...]:
    composition = formula.composition()
    return tuple(
        int(composition[symbol].count) if symbol in composition else 0
        for symbol in symbols
    )


def canonicalize_results(results, symbols: list[str]) -> list[dict]:
    records = []
    for cand in results:
        isotope_match = cand.isotope_match_result
        records.append(
            {
                "formula": cand.formula.formula,
                "counts": _counts_for_symbols(cand.formula, symbols),
                "exact_mass": cand.formula.monoisotopic_mass,
                "error_da": cand.error_da,
                "error_ppm": cand.error_ppm,
                "rdbe": cand.rdbe,
                "isotope_rmse": (
                    None if isotope_match is None else isotope_match.intensity_rmse
                ),
                "isotope_match_fraction": (
                    None if isotope_match is None else isotope_match.match_fraction
                ),
                "isotope_num_peaks_matched": (
                    None if isotope_match is None else isotope_match.num_peaks_matched
                ),
                "isotope_num_peaks_total": (
                    None if isotope_match is None else isotope_match.num_peaks_total
                ),
            }
        )
    return records


def assert_records_close(py_records: list[dict], rust_records: list[dict]) -> None:
    assert len(py_records) == len(rust_records)

    for i, (py, rs) in enumerate(zip(py_records, rust_records)):
        assert py["formula"] == rs["formula"], (
            f"formula mismatch at {i}: {py} != {rs}"
        )
        assert py["counts"] == rs["counts"], (
            f"count mismatch at {i}: {py} != {rs}"
        )
        assert abs(py["exact_mass"] - rs["exact_mass"]) < 1e-8
        assert abs(py["error_da"] - rs["error_da"]) < 1e-8
        assert abs(py["error_ppm"] - rs["error_ppm"]) < 1e-5

        if py["rdbe"] is None or rs["rdbe"] is None:
            assert py["rdbe"] is rs["rdbe"]
        else:
            assert abs(py["rdbe"] - rs["rdbe"]) < 1e-8

        for key in (
            "isotope_rmse",
            "isotope_match_fraction",
            "isotope_num_peaks_matched",
            "isotope_num_peaks_total",
        ):
            if py[key] is None or rs[key] is None:
                assert py[key] is rs[key]
            elif isinstance(py[key], float):
                assert abs(py[key] - rs[key]) < 1e-8
            else:
                assert py[key] == rs[key]


def assert_rust_matches_python(config: dict) -> None:
    config = dict(config)
    elements = config.pop("elements", "CHNOPS")

    py_finder = _finder(elements, "python")
    rust_finder = _finder(elements, "rust")

    py_results = py_finder.find_formulae(**config)
    rust_results = rust_finder.find_formulae(**config)

    symbols = list(py_finder.decomposer.element_symbols)
    py_records = canonicalize_results(py_results, symbols)
    rust_records = canonicalize_results(rust_results, symbols)

    try:
        assert_records_close(py_records, rust_records)
    except AssertionError as exc:
        py_set = {(r["formula"], r["counts"]) for r in py_records}
        rs_set = {(r["formula"], r["counts"]) for r in rust_records}

        only_py = sorted(py_set - rs_set)[:20]
        only_rs = sorted(rs_set - py_set)[:20]
        first_ordering_mismatch = None
        for i, pair in enumerate(zip(py_records, rust_records)):
            if pair[0]["formula"] != pair[1]["formula"]:
                first_ordering_mismatch = (i, pair[0], pair[1])
                break

        raise AssertionError(
            "Rust/Python mismatch\n"
            f"config={config | {'elements': elements}}\n"
            f"python_count={len(py_records)}\n"
            f"rust_count={len(rust_records)}\n"
            f"only_python={only_py}\n"
            f"only_rust={only_rs}\n"
            f"first_ordering_mismatch={first_ordering_mismatch}"
        ) from exc


def test_rust_extension_smoke():
    import find_mfs._rust as rust

    assert rust.format_formula(["H", "C", "O"], [12, 6, 6], 0) == "C6H12O6"
    assert rust.parse_formula_counts("C6H12O6", ["C", "H", "N", "O"]) == [
        6,
        12,
        0,
        6,
    ]
    assert rust.parse_element_symbols("C6H12O6") == ["C", "H", "O"]

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=180.063,
        error_ppm=5.0,
        max_results=50,
    )
    assert len(results) > 0


def test_rust_backend_bypasses_cython_decompose_and_pipeline(monkeypatch):
    from find_mfs.core.decomposer import MassDecomposer
    from find_mfs.core import _pipeline

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should not use the Cython query path")

    monkeypatch.setattr(MassDecomposer, "decompose_and_score", fail)
    monkeypatch.setattr(_pipeline, "run_query_pipeline", fail)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=250.0,
        error_da=0.01,
        filter_rdbe=(0, 30),
        check_octet=True,
        max_results=1000,
    )
    assert len(results) > 0


def test_rust_backend_does_not_build_python_mass_decomposer(monkeypatch):
    from find_mfs.core.decomposer import MassDecomposer

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should build ERT state in Rust")

    monkeypatch.setattr(MassDecomposer, "__init__", fail)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=180.063,
        error_ppm=5.0,
        max_results=10,
    )

    assert len(results) > 0


def test_rust_backend_parses_element_string_without_molmass_formula(monkeypatch):
    import molmass

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should parse element strings in Rust")

    monkeypatch.setattr(molmass, "Formula", fail)

    results = FormulaFinder("C6H12O6", backend="rust").find_formulae(
        mass=180.063,
        error_ppm=5.0,
        max_results=10,
    )

    assert len(results) > 0


def test_rust_backend_derives_setup_coefficients_without_python_helpers(monkeypatch):
    import find_mfs.core.decomposer as decomposer_module
    import find_mfs.isotopes.ratios as ratios_module

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should derive setup coefficients in Rust")

    monkeypatch.setattr(decomposer_module, "get_element_most_abundant_mass", fail)
    monkeypatch.setattr(ratios_module, "get_m1_ratio", fail)
    monkeypatch.setattr(ratios_module, "get_m2_direct", fail)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=180.063,
        error_ppm=5.0,
        max_results=10,
    )

    assert len(results) > 0


def test_rust_backend_uses_embedded_setup_tables(monkeypatch):
    import molmass.elements as molmass_elements
    from IsoSpecPy import PeriodicTbl
    from find_mfs.isotopes import _isospec_bridge

    class ExplodingIterable:
        def __iter__(self):
            raise AssertionError("rust backend should use embedded element data")

    class ExplodingMapping(dict):
        def __iter__(self):
            raise AssertionError("rust backend should use embedded isotope data")

        def keys(self):
            raise AssertionError("rust backend should use embedded isotope data")

        def items(self):
            raise AssertionError("rust backend should use embedded isotope data")

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should use embedded isotope arrays")

    monkeypatch.setattr(molmass_elements, "ELEMENTS", ExplodingIterable())
    monkeypatch.setattr(PeriodicTbl, "symbol_to_masses", ExplodingMapping())
    monkeypatch.setattr(_isospec_bridge, "get_isotope_arrays", fail)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=180.063,
        error_ppm=5.0,
        filter_rdbe=(0, 20),
        check_octet=True,
        max_results=10,
    )

    assert len(results) > 0


def test_rust_backend_parses_adducts_and_bounds_without_python_helpers(monkeypatch):
    import find_mfs.core.finder as finder_module
    from find_mfs.core.rust_backend import RustFormulaFinder

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should prepare public query inputs in Rust")

    monkeypatch.setattr(FormulaFinder, "_parse_adduct", fail)
    monkeypatch.setattr(finder_module, "to_bounds_dict", fail)
    monkeypatch.setattr(RustFormulaFinder, "parse_adduct", fail)
    monkeypatch.setattr(RustFormulaFinder, "_count_vectors", fail)

    mass = Formula("C6H12O6Na+").monoisotopic_mass
    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=mass,
        charge=1,
        adduct="Na",
        error_ppm=5.0,
        min_counts="C6H12O6",
        max_counts="C6H12N0O6P0S0",
        filter_rdbe=(0, 20),
        check_octet=True,
        max_results=20,
    )

    assert [candidate.formula.formula for candidate in results] == ["C6H12O6"]


def test_rust_backend_keeps_adduct_metadata_without_python_zip(monkeypatch):
    import find_mfs.core.rust_backend as rust_backend_module

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should keep Rust adduct vectors")

    monkeypatch.setattr(rust_backend_module, "zip", fail, raising=False)

    formula = Formula("C6H12O6")
    ion_formula = Formula("C6H12O6Na+")
    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=ion_formula.monoisotopic_mass,
        charge=1,
        adduct="Na",
        error_ppm=5.0,
        min_counts="C6H12O6",
        max_counts="C6H12O6",
        max_results=10,
    )

    assert results._backend._adduct_elements == (["Na"], [1])
    assert results[0].formula.formula == formula.formula

    predicted = results._backend._isotope_envelope_backend.simulate_isotope_envelope(
        [6, 12, 0, 6, 0, 0],
        results._backend._adduct_elements,
        1,
        0.05,
        0.001,
    )
    assert predicted.shape[0] > 0


def test_rust_backend_lazy_result_operations_stay_in_rust(monkeypatch):
    import find_mfs.core.results as results_module

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=250.0,
        error_da=0.02,
        filter_rdbe=(0, 30),
        check_octet=True,
        max_results=1000,
    )
    assert results._backend._rust_result is not None
    original_len = len(results)

    class ExplodingLightFormula:
        @staticmethod
        def from_counts(*args, **kwargs):
            raise AssertionError("lazy Rust result operations should stay in Rust")

    monkeypatch.setattr(results_module, "LightFormula", ExplodingLightFormula)

    sorted_results = results.sort_by_error()
    assert len(sorted_results) == original_len
    assert sorted_results._backend._rust_result is not None

    filtered_error = results.filter_by_error(max_da=0.01)
    assert len(filtered_error) <= original_len
    assert filtered_error._backend._rust_result is not None

    filtered_rdbe = results.filter_by_rdbe(0, 10)
    assert len(filtered_rdbe) <= original_len
    assert filtered_rdbe._backend._rust_result is not None

    filtered_octet = results.filter_by_octet()
    assert len(filtered_octet) <= original_len
    assert filtered_octet._backend._rust_result is not None


def test_rust_backend_display_and_dataframe_stay_in_rust(monkeypatch):
    pytest.importorskip("pandas")
    import find_mfs.core.results as results_module

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=250.0,
        error_da=0.02,
        filter_rdbe=(0, 30),
        check_octet=True,
        max_results=1000,
    )
    assert results._backend._rust_result is not None
    expected_formulas = results._backend._rust_result.formula_strings()

    class ExplodingLightFormula:
        @staticmethod
        def from_counts(*args, **kwargs):
            raise AssertionError("Rust display/export should use raw Rust rows")

    monkeypatch.setattr(results_module, "LightFormula", ExplodingLightFormula)

    table = results.to_table(max_rows=3)
    assert "Formula" in table
    assert "Error (Da)" in table
    assert repr(results).startswith("FormulaSearchResults(")

    df = results.to_dataframe()
    assert len(df) == len(results)
    assert df["formula"].tolist() == expected_formulas
    assert list(df.columns[:4]) == ["formula", "error_ppm", "error_da", "rdbe"]
    assert results._backend._cache == {}


def test_rust_backend_prior_scoring_stays_in_rust(monkeypatch):
    pytest.importorskip("pandas")
    import find_mfs.core.results as results_module

    prior = FormulaPrior().fit(METABOLITE_CORPUS)
    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=250.0,
        error_da=0.02,
        filter_rdbe=(0, 30),
        check_octet=True,
        max_results=1000,
    )
    assert results._backend._rust_result is not None

    class ExplodingLightFormula:
        @staticmethod
        def from_counts(*args, **kwargs):
            raise AssertionError("Rust prior scoring should stay in Rust")

    monkeypatch.setattr(results_module, "LightFormula", ExplodingLightFormula)

    ret = prior.score_results(
        results,
        mass_sigma_ppm=2.0,
        isotope_sigma=0.05,
    )
    assert ret is None
    assert results._backend._rust_result is not None
    assert results._backend._cache == {}

    sorted_prior = results.sort_by_prior()
    sorted_posterior = results.sort_by_posterior()
    assert len(sorted_prior) == len(results)
    assert len(sorted_posterior) == len(results)
    assert sorted_prior._backend._rust_result is not None
    assert sorted_posterior._backend._rust_result is not None

    table = results.to_table(max_rows=3)
    assert "Prior" in table
    df = results.to_dataframe()
    assert "prior_score" in df.columns
    assert results._backend._cache == {}


def test_rust_prior_scores_match_python_materialized_scores():
    config = {
        "mass": 250.0,
        "error_da": 0.02,
        "filter_rdbe": (0, 30),
        "check_octet": True,
        "max_results": 1000,
    }
    py_results = FormulaFinder("CHNOPS", backend="python").find_formulae(**config)
    rust_results = FormulaFinder("CHNOPS", backend="rust").find_formulae(**config)

    prior = FormulaPrior().fit(METABOLITE_CORPUS)
    prior.score_results(py_results, mass_sigma_ppm=2.0, isotope_sigma=0.05)
    prior.score_results(rust_results, mass_sigma_ppm=2.0, isotope_sigma=0.05)

    py_sorted_prior = [c.formula.formula for c in py_results.sort_by_prior()]
    rust_sorted_prior = [c.formula.formula for c in rust_results.sort_by_prior()]
    assert rust_sorted_prior == py_sorted_prior

    py_sorted_posterior = [
        c.formula.formula for c in py_results.sort_by_posterior()
    ]
    rust_sorted_posterior = [
        c.formula.formula for c in rust_results.sort_by_posterior()
    ]
    assert rust_sorted_posterior == py_sorted_posterior

    py_scores = [
        (c.formula.formula, c.prior_score, c.posterior_score)
        for c in py_results
    ]
    rust_scores = [
        (c.formula.formula, c.prior_score, c.posterior_score)
        for c in rust_results
    ]
    assert len(rust_scores) == len(py_scores)
    for py_row, rust_row in zip(py_scores, rust_scores):
        assert rust_row[0] == py_row[0]
        assert rust_row[1] == pytest.approx(py_row[1], abs=1e-8)
        assert rust_row[2] == pytest.approx(py_row[2], abs=1e-8)


def test_rust_backend_filter_by_octet_stays_in_rust(monkeypatch):
    import find_mfs.core.results as results_module

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=250.0,
        error_da=0.02,
        filter_rdbe=(0, 30),
        max_results=1000,
    )
    assert results._backend._rust_result is not None

    class ExplodingLightFormula:
        @staticmethod
        def from_counts(*args, **kwargs):
            raise AssertionError("Rust octet filtering should stay in Rust")

    monkeypatch.setattr(results_module, "LightFormula", ExplodingLightFormula)

    filtered = results.filter_by_octet()
    assert len(filtered) <= len(results)
    assert filtered._backend._rust_result is not None
    assert results._backend._cache == {}


def test_rust_backend_extracts_dict_bounds_without_python_iteration():
    class ExplodingDict(dict):
        def __iter__(self):
            raise AssertionError("rust backend should extract dict bounds in Rust")

        def keys(self):
            raise AssertionError("rust backend should extract dict bounds in Rust")

        def items(self):
            raise AssertionError("rust backend should extract dict bounds in Rust")

    glucose_counts = ExplodingDict(
        {"C": 6, "H": 12, "N": 0, "O": 6, "P": 0, "S": 0}
    )

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=Formula("C6H12O6").monoisotopic_mass,
        error_da=1e-6,
        min_counts=glucose_counts,
        max_counts=glucose_counts,
        filter_rdbe=(0, 20),
        check_octet=True,
        max_results=10,
    )

    assert [candidate.formula.formula for candidate in results] == ["C6H12O6"]


def test_rust_backend_materializes_rust_formula_strings(monkeypatch):
    import find_mfs.core.results as results_module

    original_light_formula = results_module.LightFormula
    seen_formula_strings = []

    class SpyLightFormula:
        @staticmethod
        def from_counts(
            symbols,
            counts,
            charge=0,
            monoisotopic_mass=0.0,
            formula_str=None,
        ):
            assert formula_str is not None
            seen_formula_strings.append(formula_str)
            return original_light_formula.from_counts(
                symbols=symbols,
                counts=counts,
                charge=charge,
                monoisotopic_mass=monoisotopic_mass,
                formula_str=formula_str,
            )

    monkeypatch.setattr(results_module, "LightFormula", SpyLightFormula)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=Formula("C6H12O6").monoisotopic_mass,
        error_da=1e-6,
        min_counts="C6H12O6",
        max_counts="C6H12O6",
        max_results=10,
    )

    assert results._backend._formula_strings is None
    assert results[0].formula.formula == "C6H12O6"
    assert seen_formula_strings == ["C6H12O6"]


def test_rust_backend_owns_rdbe_constants_for_rust_queries():
    finder = FormulaFinder("CHNOPS", backend="rust")
    finder._has_known_bond_e = False
    finder._unknown_bond_e_indices = np.arange(len(finder.decomposer.element_symbols))
    finder._rdbe_coeffs = np.full(len(finder.decomposer.element_symbols), 999.0)
    finder._rdbe_coeffs_fallback = np.full(
        len(finder.decomposer.element_symbols),
        999.0,
    )

    results = finder.find_formulae(
        mass=Formula("C6H12O6").monoisotopic_mass,
        error_da=1e-6,
        min_counts="C6H12O6",
        max_counts="C6H12O6",
        filter_rdbe=(0, 10),
        check_octet=True,
        max_results=10,
    )

    assert [candidate.formula.formula for candidate in results] == ["C6H12O6"]
    assert results[0].rdbe == 1.0


def test_rust_backend_owns_isotope_prefilter_ratio_extraction(monkeypatch):
    import find_mfs.core.finder as finder_module
    from find_mfs.isotopes import _isospec_bridge

    formula = Formula("C6H12O6")
    envelope = find_mfs.get_isotope_envelope(
        formula,
        mz_tolerance=0.05,
        threshold=0.001,
    )
    isotope_match = find_mfs.SingleEnvelopeMatch(
        envelope=envelope,
        mz_tolerance_da=0.01,
        minimum_rmse=0.03,
        enable_approx_prefilter=True,
    )

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should own isotope query setup")

    finder = FormulaFinder("CHNOPS", backend="rust")
    finder._get_rust_finder()
    monkeypatch.setattr(finder_module.np, "argmin", fail)
    monkeypatch.setattr(_isospec_bridge, "get_isotope_arrays", fail)

    results = finder.find_formulae(
        mass=formula.monoisotopic_mass,
        error_ppm=5.0,
        min_counts="C6H12O6",
        max_counts="C6H12N0O6P0S0",
        isotope_match=isotope_match,
        max_results=1000,
    )

    assert [candidate.formula.formula for candidate in results] == ["C6H12O6"]


def test_rust_backend_reuses_stored_finder_state():
    finder = FormulaFinder("CHNOPS", backend="rust")
    assert finder._rust_finder is None

    first = finder.find_formulae(180.063, error_ppm=5.0, max_results=20)
    rust_finder = finder._rust_finder
    second = finder.find_formulae(250.0, error_da=0.01, max_results=20)

    assert len(first) > 0
    assert len(second) > 0
    assert finder._rust_finder is rust_finder


def test_rust_negative_max_results_raises_value_error_like_python():
    config = {
        "mass": 180.063,
        "error_da": 0.01,
        "max_results": -1,
    }

    with pytest.raises(ValueError):
        FormulaFinder("CHNOPS", backend="python").find_formulae(**config)

    with pytest.raises(ValueError, match="max_results must be non-negative"):
        FormulaFinder("CHNOPS", backend="rust").find_formulae(**config)


@pytest.mark.parametrize(
    "config",
    [
        {"mass": 180.063, "error_ppm": float("nan"), "error_da": 0.01},
        {"mass": float("inf"), "error_da": 0.01},
    ],
)
def test_rust_nonfinite_mass_window_matches_python_empty_results(config: dict):
    config = {**config, "max_results": 10}

    py_results = FormulaFinder("CHNOPS", backend="python").find_formulae(**config)
    rust_results = FormulaFinder("CHNOPS", backend="rust").find_formulae(**config)

    assert len(py_results) == 0
    assert len(rust_results) == 0


@pytest.mark.parametrize(
    ("formula", "symbols", "expected_counts", "expected_formatted"),
    [
        ("H2O", ["C", "H", "O"], [0, 2, 1], "H2O"),
        ("CO2", ["C", "H", "O"], [1, 0, 2], "CO2"),
        ("C6H12O6", ["C", "H", "O"], [6, 12, 6], "C6H12O6"),
        ("OH2", ["C", "H", "O"], [0, 2, 1], "H2O"),
        ("C20H40P0", ["C", "H", "N", "O", "P"], [20, 40, 0, 0, 0], "C20H40"),
    ],
)
def test_rust_formula_parse_and_format(
    formula: str,
    symbols: list[str],
    expected_counts: list[int],
    expected_formatted: str,
):
    import find_mfs._rust as rust

    counts = rust.parse_formula_counts(formula, symbols)
    assert counts == expected_counts
    assert rust.format_formula(symbols, counts, 0) == expected_formatted


def test_rust_finder_wrapper_parses_bounds_and_adducts():
    finder = FormulaFinder("CHNOPS", backend="rust")._get_rust_finder()

    min_values, max_values = finder._count_vectors("C5O*", "C10H20N*S0P0")
    symbols = finder._symbols
    assert min_values[symbols.index("C")] == 5
    assert min_values[symbols.index("O")] == 0
    assert max_values[symbols.index("N")] == float("inf")
    assert max_values[symbols.index("S")] == 0.0
    assert max_values[symbols.index("P")] == 0.0

    mass, adduct_elements = finder.parse_adduct("-H")
    assert mass < 0
    assert adduct_elements == {"H": -1}


@pytest.mark.parametrize("bad_formula", ["Xx2", "C6Q", "2H", "C6-H12"])
def test_rust_formula_parse_invalid(bad_formula: str):
    import find_mfs._rust as rust

    with pytest.raises(ValueError):
        rust.parse_formula_counts(bad_formula, ["C", "H", "N", "O"])


@pytest.mark.parametrize(
    "config",
    [
        {
            "elements": "CHNOPS",
            "mass": Formula("H2O").monoisotopic_mass,
            "error_da": 1e-6,
            "max_results": 20,
        },
        {
            "elements": "CHO",
            "mass": Formula("CO2").monoisotopic_mass,
            "error_da": 1e-6,
            "max_results": 20,
        },
        {
            "elements": "CHNOPS",
            "mass": 180.063,
            "error_ppm": 5.0,
            "filter_rdbe": (0, 20),
            "check_octet": True,
            "max_results": 500,
        },
        {
            "elements": "CHNOPS",
            "mass": 180.063,
            "error_da": 0.01,
            "max_results": 5,
        },
        {
            "elements": "CHNOPS",
            "mass": 250.0,
            "error_ppm": 10.0,
            "max_counts": {"P": 0, "S": 0},
            "max_results": 1000,
        },
    ],
)
def test_rust_matches_python_deterministic_configs(config: dict):
    assert_rust_matches_python(config)


@pytest.mark.parametrize("mass", [18.010565, 44.0095, 100.0, 150.0, 250.0, 500.0, 750.0, 1000.0, 1500.0])
@pytest.mark.parametrize(
    "tolerance",
    [
        {"error_ppm": 1.0},
        {"error_ppm": 5.0},
        {"error_ppm": 10.0},
        {"error_ppm": 20.0},
        {"error_da": 0.001},
        {"error_da": 0.005},
        {"error_da": 0.01},
    ],
)
def test_rust_matches_python_diverse_mass_tolerance_grid(
    mass: float,
    tolerance: dict,
):
    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": mass,
            "max_results": 1000,
            **tolerance,
        }
    )


@pytest.mark.parametrize(
    "config",
    [
        {"min_counts": {"C": 1}},
        {"max_counts": {"C": 10}},
        {"min_counts": {"O": 1}},
        {"max_counts": {"P": 0}},
        {"max_counts": {"S": 0}},
        {"min_counts": {"N": 0}, "max_counts": {"N": 0}},
        {"min_counts": {"P": 1}, "max_counts": {"P": 1}},
        {"min_counts": "C5", "max_counts": "C10H20N*S0P0"},
    ],
)
def test_rust_matches_python_constraints(config: dict):
    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": 180.063,
            "error_da": 0.02,
            "max_results": 1000,
            **config,
        }
    )


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"filter_rdbe": (0, 20)},
        {"check_octet": True},
        {"filter_rdbe": (0, 20), "check_octet": True},
    ],
)
def test_rust_matches_python_filter_combinations(filters: dict):
    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": 250.0,
            "error_da": 0.02,
            "max_results": 1000,
            **filters,
        }
    )


@pytest.mark.parametrize(
    ("elements", "mass"),
    [
        ("CH", 100.0),
        ("CHO", 150.0),
        ("CHNOPS", 250.0),
        ("CHNOPSCl", 250.0),
        ("CHNOPSClBr", 300.0),
    ],
)
def test_rust_matches_python_element_sets(elements: str, mass: float):
    assert_rust_matches_python(
        {
            "elements": elements,
            "mass": mass,
            "error_da": 0.01,
            "max_results": 1000,
        }
    )


@pytest.mark.parametrize(
    "config",
    [
        {"mass": 0.0, "error_da": 0.001},
        {"mass": -1.0, "error_da": 0.001},
        {"mass": 1.0, "error_da": 0.0001},
        {"mass": 180.063, "error_ppm": 0.000001},
        {"mass": 180.063, "error_da": 0.01, "max_results": 0},
        {"mass": 180.063, "error_da": 0.01, "max_results": 1},
        {
            "mass": 180.063,
            "error_da": 0.01,
            "min_counts": {"C": 10},
            "max_counts": {"C": 5},
        },
        {
            "mass": 180.063,
            "error_da": 0.01,
            "min_counts": {"C": 100},
            "max_counts": {"H": 1},
        },
    ],
)
def test_rust_matches_python_edge_cases(config: dict):
    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "max_results": 1000,
            **config,
        }
    )


def test_rust_matches_python_charge_and_adduct():
    glucose_counts = {"C": 6, "H": 12, "N": 0, "O": 6, "P": 0, "S": 0}
    sodium_adduct_mass = Formula("C6H12O6Na+").monoisotopic_mass

    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": sodium_adduct_mass,
            "charge": 1,
            "adduct": "Na",
            "error_ppm": 5.0,
            "min_counts": glucose_counts,
            "max_counts": glucose_counts,
            "filter_rdbe": (0, 20),
            "check_octet": True,
            "max_results": 20,
        }
    )

    deprotonated_mass = Formula("C6H11O6-").monoisotopic_mass
    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": deprotonated_mass,
            "charge": -1,
            "adduct": "-H",
            "error_ppm": 5.0,
            "min_counts": glucose_counts,
            "max_counts": glucose_counts,
            "filter_rdbe": (0, 20),
            "check_octet": True,
            "max_results": 20,
        }
    )


def test_rust_matches_python_posthoc_octet_filter():
    config = {
        "mass": 250.0,
        "error_da": 0.02,
        "filter_rdbe": (0, 30),
        "max_results": 1000,
    }
    py_results = FormulaFinder("CHNOPS", backend="python").find_formulae(**config)
    rust_results = FormulaFinder("CHNOPS", backend="rust").find_formulae(**config)

    symbols = list(FormulaFinder("CHNOPS").decomposer.element_symbols)
    assert_records_close(
        canonicalize_results(py_results.filter_by_octet(), symbols),
        canonicalize_results(rust_results.filter_by_octet(), symbols),
    )


def test_rust_matches_python_with_isotope_prefilter():
    formula = Formula("C6H12O6")
    envelope = find_mfs.get_isotope_envelope(
        formula,
        mz_tolerance=0.05,
        threshold=0.001,
    )
    isotope_match = find_mfs.SingleEnvelopeMatch(
        envelope=envelope,
        mz_tolerance_da=0.01,
        minimum_rmse=0.03,
        enable_approx_prefilter=True,
    )

    assert_rust_matches_python(
        {
            "elements": "CHNOPS",
            "mass": formula.monoisotopic_mass,
            "error_ppm": 5.0,
            "max_counts": {"C": 10, "H": 20, "N": 4, "O": 10, "P": 1, "S": 1},
            "isotope_match": isotope_match,
            "max_results": 1000,
        }
    )


def test_rust_backend_materializes_predicted_envelope_without_python_helper(monkeypatch):
    import find_mfs.isotopes.envelope as envelope_module

    formula = Formula("C6H12O6")
    envelope = find_mfs.get_isotope_envelope(
        formula,
        mz_tolerance=0.05,
        threshold=0.001,
    )
    isotope_match = find_mfs.SingleEnvelopeMatch(
        envelope=envelope,
        mz_tolerance_da=0.01,
        minimum_rmse=0.03,
        enable_approx_prefilter=True,
    )

    def fail(*args, **kwargs):
        raise AssertionError("rust backend should materialize isotope envelopes in Rust")

    monkeypatch.setattr(envelope_module, "get_isotope_envelope", fail)

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=formula.monoisotopic_mass,
        error_da=1e-8,
        min_counts="C6H12O6",
        max_counts="C6H12O6",
        isotope_match=isotope_match,
        max_results=10,
    )
    isotope_result = results[0].isotope_match_result

    assert isotope_result is not None
    assert isotope_result.predicted_envelope.ndim == 2
    assert isotope_result.predicted_envelope.shape[1] == 2
    assert isotope_result.predicted_envelope.shape[0] > 0


def test_rust_backend_extracts_envelope_adduct_dict_in_rust():
    class ExplodingDict(dict):
        def __iter__(self):
            raise AssertionError("rust backend should extract adduct dicts in Rust")

        def keys(self):
            raise AssertionError("rust backend should extract adduct dicts in Rust")

        def items(self):
            raise AssertionError("rust backend should extract adduct dicts in Rust")

    finder = FormulaFinder("CHNOPS", backend="rust")._get_rust_finder()
    envelope = finder.simulate_isotope_envelope(
        core_counts=[6, 12, 0, 6, 0, 0],
        adduct_elements=ExplodingDict({"Na": 1}),
        charge=1,
        mz_tolerance=0.05,
        threshold=0.001,
    )

    assert envelope.ndim == 2
    assert envelope.shape[1] == 2
    assert envelope.shape[0] > 0


def test_rust_backend_extracts_isotope_config_without_python_numpy_coercion():
    class RustOnlyEnvelope:
        def __init__(self, rows):
            self._rows = rows

        def tolist(self):
            return self._rows

        def __array__(self, *args, **kwargs):
            raise AssertionError("rust backend should extract isotope envelopes in Rust")

        @property
        def shape(self):
            raise AssertionError("rust backend should not inspect envelope shape in Python")

    class RustOnlyIsotopeMatch:
        def __init__(self, rows):
            self.envelope = RustOnlyEnvelope(rows)
            self.mz_tolerance_da = 0.01
            self.mz_tolerance_ppm = 0.0
            self.simulated_mz_tolerance = 0.05
            self.simulated_intensity_threshold = 0.001
            self.minimum_rmse = 0.03
            self.enable_approx_prefilter = True
            self.approx_tolerance_rel = 0.5
            self.approx_tolerance_abs = 0.3

    formula = Formula("C6H12O6")
    envelope = find_mfs.get_isotope_envelope(
        formula,
        mz_tolerance=0.05,
        threshold=0.001,
    )
    isotope_match = RustOnlyIsotopeMatch(envelope.tolist())

    results = FormulaFinder("CHNOPS", backend="rust").find_formulae(
        mass=formula.monoisotopic_mass,
        error_da=1e-8,
        min_counts="C6H12O6",
        max_counts="C6H12O6",
        isotope_match=isotope_match,
        max_results=10,
    )

    isotope_result = results[0].isotope_match_result
    assert isotope_result is not None
    assert isotope_result.num_peaks_total == len(envelope)
    assert isotope_result.num_peaks_matched > 0


def test_rust_matches_python_randomized_small_suite():
    rng = random.Random(20260520)
    element_sets = ["CH", "CHO", "CHNO", "CHNOPS"]

    for _ in range(100):
        elements = rng.choice(element_sets)
        config = {
            "elements": elements,
            "mass": rng.uniform(50.0, 350.0),
            "max_results": rng.choice([50, 100, 250, 500]),
        }
        if rng.random() < 0.5:
            config["error_ppm"] = rng.choice([1.0, 5.0, 10.0, 20.0])
        else:
            config["error_da"] = rng.choice([0.001, 0.005, 0.01])

        if rng.random() < 0.4:
            max_counts = {}
            for symbol in FormulaFinder(elements).decomposer.element_symbols:
                if symbol == "H":
                    max_counts[symbol] = rng.randint(4, 80)
                elif symbol == "C":
                    max_counts[symbol] = rng.randint(1, 40)
                else:
                    max_counts[symbol] = rng.randint(0, 12)
            config["max_counts"] = max_counts

        assert_rust_matches_python(config)
