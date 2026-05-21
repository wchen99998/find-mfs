"""
Rust backend bridge for mass decomposition.

This module keeps the Python public API connected to the Rust-owned finder
state. Per-query decomposition, filtering, isotope scoring, and raw result
storage live in the private PyO3 extension.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional

import numpy as np
from molmass.elements import ELECTRON

if TYPE_CHECKING:
    from .decomposer import MassDecomposer
    from ..isotopes.config import IsotopeMatchConfig

AdductElements = dict[str, int] | tuple[list[str], list[int]]


def _load_rust_extension():
    try:
        return importlib.import_module("find_mfs._rust")
    except ImportError as exc:
        raise ImportError(
            "Rust backend requested, but find_mfs._rust is not installed. "
            "Build it with: uv run maturin develop --manifest-path "
            "find_mfs/rust/Cargo.toml"
        ) from exc


class RustFormulaFinder:
    """
    Python wrapper around the Rust-owned finder state.

    The heavy, element-set-specific arrays are copied into Rust once at
    construction. Per-query calls pass only scalar options, count vectors, and
    optional isotope inputs.
    """

    def __init__(self, decomposer: "MassDecomposer"):
        rust = _load_rust_extension()
        self._decomposer = decomposer
        self._symbols = list(decomposer.element_symbols)
        self._n_elem = len(self._symbols)
        self._inner = rust.RustFormulaFinder.from_precomputed_embedded_sources(
            self._symbols,
            np.ascontiguousarray(decomposer.ERT, dtype=np.float64).tolist(),
            np.ascontiguousarray(decomposer.integer_masses, dtype=np.int64).tolist(),
            np.ascontiguousarray(decomposer.real_masses, dtype=np.float64).tolist(),
            float(decomposer.precision),
            float(decomposer.min_error),
            float(decomposer.max_error),
            self._isospec_lib_path(),
        )

    @classmethod
    def from_elements(cls, elements) -> "RustFormulaFinder":
        rust = _load_rust_extension()

        if isinstance(elements, str):
            requested_symbols = list(rust.parse_element_symbols(elements))
        else:
            requested_symbols = list(elements)

        obj = cls.__new__(cls)
        obj._decomposer = None
        obj._symbols = []
        obj._n_elem = 0
        obj._inner = rust.RustFormulaFinder.from_embedded_sources(
            requested_symbols,
            cls._isospec_lib_path(),
        )
        obj._symbols = list(obj._inner.element_symbols())
        obj._n_elem = len(obj._symbols)
        return obj

    def with_sirius_like_tables(self, scoring_tables) -> "RustFormulaFinder":
        obj = self.__class__.__new__(self.__class__)
        obj._decomposer = self._decomposer
        obj._symbols = list(self._symbols)
        obj._n_elem = self._n_elem
        obj._inner = self._inner.with_sirius_like_tables(
            scoring_tables.to_rust_payload()
        )
        return obj

    @staticmethod
    def _isospec_lib_path() -> str:
        from IsoSpecPy.isoFFI import isoFFI

        return str(isoFFI.libpath)

    def _count_vectors(
        self,
        min_counts: Optional[dict[str, int] | str],
        max_counts: Optional[dict[str, int] | str],
    ) -> tuple[list[int], list[float]]:
        if isinstance(min_counts, str):
            min_values = self._inner.parse_min_counts(min_counts)
        else:
            final_min = {symbol: 0 for symbol in self._symbols}
            if min_counts:
                for symbol, count in min_counts.items():
                    final_min[symbol] = 0 if count == float("inf") else int(count)
            min_values = [int(final_min[symbol]) for symbol in self._symbols]

        if isinstance(max_counts, str):
            max_values = self._inner.parse_max_counts(max_counts)
        else:
            final_max = {symbol: float("inf") for symbol in self._symbols}
            if max_counts:
                final_max.update(max_counts)
            max_values = [float(final_max[symbol]) for symbol in self._symbols]

        return min_values, max_values

    def parse_adduct(
        self,
        adduct: Optional[str],
    ) -> tuple[float, dict[str, int]]:
        if not adduct:
            return 0.0, {}
        adduct_mass, adduct_symbols, adduct_counts = self._inner.parse_adduct(adduct)
        return float(adduct_mass), {
            symbol: int(count)
            for symbol, count in zip(adduct_symbols, adduct_counts)
            if int(count) != 0
        }

    def simulate_isotope_envelope(
        self,
        core_counts: list[int],
        adduct_elements: Optional[AdductElements],
        charge: int,
        mz_tolerance: float,
        threshold: float,
    ) -> np.ndarray:
        mz, intensity = self._inner.simulate_isotope_envelope_python(
            [int(count) for count in core_counts],
            adduct_elements,
            int(charge),
            float(ELECTRON.mass),
            float(mz_tolerance),
            float(threshold),
        )
        if not mz:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack(
            (
                np.asarray(mz, dtype=np.float64),
                np.asarray(intensity, dtype=np.float64),
            )
        )

    def find_formulae_raw(
        self,
        mass: float,
        charge: int = 0,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int] | str] = None,
        max_counts: Optional[dict[str, int] | str] = None,
        max_results: int = 10000,
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
        adduct: Optional[str] = None,
        isotope_match: Optional["IsotopeMatchConfig"] = None,
    ) -> tuple[dict, list[str], float, AdductElements]:
        rdbe_min = filter_rdbe[0] if filter_rdbe is not None else 0.0
        rdbe_max = filter_rdbe[1] if filter_rdbe is not None else 0.0

        rust_result, adduct_info = self._inner.find_formulae_public_result_python(
            float(mass),
            int(charge),
            float(ppm_error or 0.0),
            float(mz_error or 0.0),
            min_counts,
            max_counts,
            int(max_results),
            filter_rdbe is not None,
            float(rdbe_min),
            float(rdbe_max),
            bool(check_octet),
            adduct or "",
            isotope_match,
            float(ELECTRON.mass),
        )

        raw = {"rust_result": rust_result}
        adduct_mass, adduct_symbols, adduct_counts = adduct_info
        adduct_mass = float(adduct_mass)
        adduct_elements = (list(adduct_symbols), list(adduct_counts))
        return raw, list(self._symbols), adduct_mass, adduct_elements

    def find_fragmentation_tree_from_spectrum_raw(
        self,
        *,
        precursor_mz: float,
        precursor_formula: str,
        precursor_ion: str,
        peaks,
        scoring_config,
        reduce_graph: bool,
        minimal_score: float | None,
        time_limit_seconds: float | None,
        threads: int | None,
        solver: str,
    ):
        return self._inner.find_fragmentation_tree_from_spectrum_python(
            float(precursor_mz),
            str(precursor_formula),
            str(precursor_ion),
            peaks,
            scoring_config,
            bool(reduce_graph),
            None if minimal_score is None else float(minimal_score),
            None if time_limit_seconds is None else float(time_limit_seconds),
            None if threads is None else int(threads),
            float(ELECTRON.mass),
            str(solver),
        )

    def find_fragmentation_tree_result_from_spectrum_raw(
        self,
        *,
        precursor_mz: float,
        precursor_formula: str,
        precursor_ion: str,
        peaks,
        scoring_config,
        reduce_graph: bool,
        minimal_score: float | None,
        time_limit_seconds: float | None,
        threads: int | None,
        solver: str,
    ):
        return self._inner.find_fragmentation_tree_from_spectrum_result_python(
            float(precursor_mz),
            str(precursor_formula),
            str(precursor_ion),
            peaks,
            scoring_config,
            bool(reduce_graph),
            None if minimal_score is None else float(minimal_score),
            None if time_limit_seconds is None else float(time_limit_seconds),
            None if threads is None else int(threads),
            float(ELECTRON.mass),
            str(solver),
        )
