from typing import Any, Optional

class RustQueryResult:
    def __len__(self) -> int: ...
    def n_observed(self) -> int: ...
    def formula_strings(self) -> list[str]: ...
    def table_rows(
        self,
        max_rows: Optional[int],
    ) -> list[
        tuple[
            str,
            float,
            float,
            Optional[float],
            Optional[str],
            Optional[float],
            Optional[float],
        ]
    ]: ...
    def row(
        self,
        idx: int,
    ) -> tuple[
        list[int],
        float,
        float,
        float,
        Optional[float],
        Optional[tuple[float, float, int, list[int]]],
        str,
    ]: ...
    def take_indices(self, indices: list[int]) -> "RustQueryResult": ...
    def score_values(
        self,
        idx: int,
    ) -> tuple[Optional[float], Optional[float]]: ...
    def sort_by_error(self, reverse: bool) -> "RustQueryResult": ...
    def sort_by_rmse(self, reverse: bool) -> "RustQueryResult": ...
    def sort_by_prior(self, reverse: bool) -> "RustQueryResult": ...
    def sort_by_posterior(self, reverse: bool) -> "RustQueryResult": ...
    def score_prior(
        self,
        core_symbols: list[str],
        ratio_elements: list[str],
        p_absent: list[float],
        kde_points: list[list[float]],
        kde_weights: list[list[float]],
        kde_variance: list[float],
        uniform_weight: float,
        mass_sigma_ppm: float,
        isotope_sigma: float,
    ) -> "RustQueryResult": ...
    def filter_by_rdbe(
        self,
        min_rdbe: float,
        max_rdbe: float,
    ) -> "RustQueryResult": ...
    def filter_by_error(
        self,
        max_ppm: Optional[float],
        max_da: Optional[float],
    ) -> "RustQueryResult": ...
    def filter_by_isotope_quality(
        self,
        max_match_rmse: float,
        min_match_fraction: float,
    ) -> "RustQueryResult": ...
    def filter_by_octet(self, charge: int) -> "RustQueryResult": ...

class RustFormulaFinder:
    def __init__(
        self,
        element_symbols: list[str],
        ert: list[list[float]],
        integer_masses: list[int],
        real_masses: list[float],
        precision: float,
        min_error: float,
        max_error: float,
        element_mass_symbols: list[str],
        element_mass_values: list[float],
        iso_m1_coeffs: list[float],
        iso_m2_direct: list[float],
        isospec_lib_path: str,
        isotope_table_symbols: list[str],
        isotope_numbers: list[int],
        flat_isotope_masses: list[float],
        flat_isotope_probs: list[float],
    ) -> None: ...

    @staticmethod
    def from_element_masses(
        element_symbols: list[str],
        element_masses_for_finder: list[float],
        element_mass_symbols: list[str],
        element_mass_values: list[float],
        isotope_coeff_symbols: list[str],
        isotope_m1_values: list[float],
        isotope_m2_values: list[float],
        isospec_lib_path: str,
        isotope_table_symbols: list[str],
        isotope_numbers: list[int],
        flat_isotope_masses: list[float],
        flat_isotope_probs: list[float],
    ) -> "RustFormulaFinder": ...

    @staticmethod
    def from_precomputed_sources(
        element_symbols: list[str],
        ert: list[list[float]],
        integer_masses: list[int],
        real_masses: list[float],
        precision: float,
        min_error: float,
        max_error: float,
        element_source_symbols: list[str],
        element_source_isotope_numbers: list[int],
        flat_element_mass_numbers: list[int],
        flat_element_isotope_masses: list[float],
        flat_element_isotope_abundances: list[float],
        isospec_lib_path: str,
        isotope_table_symbols: list[str],
        isotope_numbers: list[int],
        flat_isotope_masses: list[float],
        flat_isotope_probs: list[float],
    ) -> "RustFormulaFinder": ...

    @staticmethod
    def from_element_sources(
        element_symbols: list[str],
        element_source_symbols: list[str],
        element_source_isotope_numbers: list[int],
        flat_element_mass_numbers: list[int],
        flat_element_isotope_masses: list[float],
        flat_element_isotope_abundances: list[float],
        isospec_lib_path: str,
        isotope_table_symbols: list[str],
        isotope_numbers: list[int],
        flat_isotope_masses: list[float],
        flat_isotope_probs: list[float],
    ) -> "RustFormulaFinder": ...

    @staticmethod
    def from_precomputed_embedded_sources(
        element_symbols: list[str],
        ert: list[list[float]],
        integer_masses: list[int],
        real_masses: list[float],
        precision: float,
        min_error: float,
        max_error: float,
        isospec_lib_path: str,
    ) -> "RustFormulaFinder": ...

    @staticmethod
    def from_embedded_sources(
        element_symbols: list[str],
        isospec_lib_path: str,
    ) -> "RustFormulaFinder": ...

    def element_symbols(self) -> list[str]: ...

    def simulate_isotope_envelope(
        self,
        core_counts: list[int],
        adduct_symbols: list[str],
        adduct_counts: list[int],
        charge: int,
        electron_mass: float,
        simulated_mz_tolerance: float,
        simulated_intensity_threshold: float,
    ) -> tuple[list[float], list[float]]: ...

    def simulate_isotope_envelope_python(
        self,
        core_counts: list[int],
        adduct_elements: Any,
        charge: int,
        electron_mass: float,
        simulated_mz_tolerance: float,
        simulated_intensity_threshold: float,
    ) -> tuple[list[float], list[float]]: ...

    def parse_min_counts(self, formula: str) -> list[int]: ...

    def parse_max_counts(self, formula: str) -> list[float]: ...

    def parse_adduct(self, adduct: str) -> tuple[float, list[str], list[int]]: ...

    def find_formulae_public_result_python(
        self,
        mass: float,
        charge: int,
        ppm_error: float,
        mz_error: float,
        min_counts: Any,
        max_counts: Any,
        max_results: int,
        apply_rdbe_filter: bool,
        rdbe_min: float,
        rdbe_max: float,
        check_octet: bool,
        adduct: str,
        isotope_match: Any,
        electron_mass: float,
    ) -> tuple[
        RustQueryResult,
        tuple[float, list[str], list[int]],
    ]: ...

def format_formula(symbols: list[str], counts: list[int], charge: int) -> str: ...

def parse_formula_counts(formula_str: str, symbols: list[str]) -> list[int]: ...

def parse_element_symbols(formula_str: str) -> list[str]: ...
