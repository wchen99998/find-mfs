"""
This module has the FormulaSearchResults class, which contains
FormulaCandidate objects, and provides convenience methods for:
- filtering,
- display
- export
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, overload, TYPE_CHECKING

import numpy as np

from .finder import FormulaCandidate
from .light_formula import LightFormula
from ..utils.filtering import passes_octet_rule
from ..isotopes import IsotopeMatchResult
from ..utils.table import render_table, render_dataframe

if TYPE_CHECKING:
    import pandas as pd

AdductElements = dict[str, int] | tuple[list[str], list[int]]
RawDisplayRow = tuple[
    str,
    float,
    float,
    Optional[float],
    Optional[str],
    Optional[float],
    Optional[float],
]


def _raw_rows_have_isotope(rows: list[RawDisplayRow]) -> bool:
    return any(
        isotope_matches is not None or isotope_rmse is not None
        for _, _, _, _, isotope_matches, isotope_rmse, _ in rows
    )


def _raw_rows_have_prior(rows: list[RawDisplayRow]) -> bool:
    return any(prior_score is not None for *_, prior_score in rows)


def _render_raw_table(
    rows: list[RawDisplayRow],
    max_rows: Optional[int],
    total: int,
) -> str:
    if not rows:
        return "No candidates found."

    columns = [
        ('Formula', 25, '<', lambda row: row[0]),
        ('Error (ppm)', 15, '>', lambda row: f"{row[1]:.2f}"),
        ('Error (Da)', 15, '>', lambda row: f"{row[2]:.6f}"),
        (
            'RDBE',
            10,
            '>',
            lambda row: f"{row[3]:.1f}" if row[3] is not None else "N/A",
        ),
    ]
    if _raw_rows_have_isotope(rows):
        columns.extend(
            [
                ('Iso. Matches', 15, '>', lambda row: row[4] or ""),
                (
                    'Iso. RMSE',
                    10,
                    '>',
                    lambda row: (
                        f"{row[5]:.4f}" if row[5] is not None else ""
                    ),
                ),
            ]
        )
    if _raw_rows_have_prior(rows):
        columns.append(
            (
                'Prior',
                10,
                '>',
                lambda row: (
                    f"{row[6]:.2f}" if row[6] is not None else ""
                ),
            )
        )

    header = " ".join(
        f"{title:{align}{width}}" for title, width, align, _ in columns
    )
    sep = "-" * len(header)
    body = [
        " ".join(
            f"{value(row):{align}{width}}"
            for _, width, align, value in columns
        )
        for row in rows
    ]
    lines = [header, sep] + body
    if max_rows is not None and total > max_rows:
        lines.append(f"... and {total - max_rows} more")
    return "\n".join(lines)


def _render_raw_dataframe(rows: list[RawDisplayRow]) -> 'pd.DataFrame':
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install with: pip install pandas"
        )

    has_isotope = _raw_rows_have_isotope(rows)
    has_prior = _raw_rows_have_prior(rows)
    data = []
    for (
        formula,
        error_ppm,
        error_da,
        rdbe,
        isotope_matches,
        isotope_rmse,
        prior_score,
    ) in rows:
        row = {
            'formula': formula,
            'error_ppm': error_ppm,
            'error_da': error_da,
            'rdbe': rdbe,
        }
        if has_isotope:
            row['isotope_matches'] = isotope_matches
            row['isotope_rmse'] = isotope_rmse
        if has_prior:
            row['prior_score'] = prior_score
        data.append(row)

    return pd.DataFrame(data)


class _LazyBackend:
    """
    Stores raw numpy arrays and materializes FormulaCandidate on demand.

    This avoids eagerly constructing N LightFormula + N FormulaCandidate +
    N SingleEnvelopeMatchResult Python objects when the user may only
    inspect a few of them.
    """
    __slots__ = (
        '_rust_result',
        '_counts', '_exact_masses', '_error_ppm', '_error_da',
        '_rdbe', '_iso_rmse', '_iso_match_frac', '_iso_n_matched',
        '_iso_peak_matches', '_formula_strings',
        '_symbols', '_charge', '_ion_charge', '_adduct', '_adduct_elements', '_n_obs',
        '_charge_mass_offset', '_adduct_mass',
        '_simulated_mz_tolerance', '_simulated_intensity_threshold',
        '_isotope_envelope_backend',
        '_cache',
    )

    def __init__(
        self,
        raw: dict,
        symbols: list[str],
        charge: int,
        ion_charge: int,
        adduct: str | None = None,
        adduct_elements: AdductElements | None = None,
        n_obs: int = 0,
        charge_mass_offset: float = 0.0,
        adduct_mass: float = 0.0,
        simulated_mz_tolerance: float | None = None,
        simulated_intensity_threshold: float | None = None,
        isotope_envelope_backend=None,
    ):
        self._rust_result = raw.get('rust_result')
        if self._rust_result is None:
            self._counts = raw['counts']
            self._exact_masses = raw['exact_masses']
            self._error_ppm = raw['error_ppm']
            self._error_da = raw['error_da']
            self._rdbe = raw.get('rdbe')
            self._iso_rmse = raw.get('iso_rmse')
            self._iso_match_frac = raw.get('iso_match_frac')
            self._iso_n_matched = raw.get('iso_n_matched')
            self._iso_peak_matches = raw.get('iso_peak_matches')
            self._formula_strings = raw.get('formula_strings')
        else:
            self._counts = None
            self._exact_masses = None
            self._error_ppm = None
            self._error_da = None
            self._rdbe = None
            self._iso_rmse = None
            self._iso_match_frac = None
            self._iso_n_matched = None
            self._iso_peak_matches = None
            self._formula_strings = None
        self._symbols = symbols
        self._charge = charge
        self._ion_charge = ion_charge
        self._adduct = adduct
        self._adduct_elements = adduct_elements
        self._n_obs = n_obs
        self._charge_mass_offset = charge_mass_offset
        self._adduct_mass = adduct_mass
        self._simulated_mz_tolerance = simulated_mz_tolerance
        self._simulated_intensity_threshold = simulated_intensity_threshold
        self._isotope_envelope_backend = isotope_envelope_backend
        self._cache: dict[int, FormulaCandidate] = {}

    def __len__(self) -> int:
        if self._rust_result is not None:
            return len(self._rust_result)
        return self._counts.shape[0]

    def _build_ion_formula(
        self,
        idx: int,
        row_list: list[int],
        core_formula: LightFormula,
    ) -> LightFormula | None:
        if self._adduct_elements is None:
            return core_formula

        ion_elements = {
            sym: count for sym, count in zip(self._symbols, row_list) if count > 0
        }
        if isinstance(self._adduct_elements, tuple):
            adduct_items = zip(*self._adduct_elements)
        else:
            adduct_items = self._adduct_elements.items()

        for sym, delta in adduct_items:
            updated = ion_elements.get(sym, 0) + delta
            if updated < 0:
                return None
            if updated == 0:
                ion_elements.pop(sym, None)
            else:
                ion_elements[sym] = updated

        return LightFormula(
            elements=ion_elements,
            charge=self._ion_charge,
            monoisotopic_mass=float(self._exact_masses[idx]),
        )

    def _materialize(self, idx: int) -> FormulaCandidate:
        if idx in self._cache:
            return self._cache[idx]

        prior_score = None
        posterior_score = None
        if self._rust_result is None:
            row_list = self._counts[idx].tolist()
            exact_mass = float(self._exact_masses[idx])
            error_ppm = float(self._error_ppm[idx])
            error_da = float(self._error_da[idx])
            rdbe = float(self._rdbe[idx]) if self._rdbe is not None else None
            formula_str = (
                None if self._formula_strings is None
                else self._formula_strings[idx]
            )
            isotope_row = None
            if self._iso_rmse is not None:
                if self._iso_peak_matches is not None:
                    peak_matches = self._iso_peak_matches[idx].astype(bool)
                else:
                    peak_matches = np.full(
                        self._n_obs,
                        self._iso_n_matched[idx] > 0,
                    )
                isotope_row = (
                    float(self._iso_rmse[idx]),
                    float(self._iso_match_frac[idx]),
                    int(self._iso_n_matched[idx]),
                    peak_matches,
                )
        else:
            (
                row_list,
                exact_mass,
                error_ppm,
                error_da,
                rdbe,
                isotope_payload,
                formula_str,
            ) = self._rust_result.row(idx)
            isotope_row = None
            if isotope_payload is not None:
                rmse, match_frac, n_matched, peak_matches = isotope_payload
                isotope_row = (
                    float(rmse),
                    float(match_frac),
                    int(n_matched),
                    np.asarray(peak_matches, dtype=bool),
                )
            prior_score, posterior_score = self._rust_result.score_values(idx)

        if self._adduct is not None:
            # Adduct path: core molecule is neutral
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=0,
                monoisotopic_mass=(
                    float(exact_mass)
                    + self._charge_mass_offset
                    - self._adduct_mass
                ),
                formula_str=formula_str,
            )
        else:
            formula = LightFormula.from_counts(
                symbols=self._symbols,
                counts=row_list,
                charge=self._charge,
                monoisotopic_mass=float(exact_mass),
                formula_str=formula_str,
            )

        isotope_result = None
        if isotope_row is not None:
            predicted_envelope = np.empty((0, 2), dtype=np.float64)
            if (
                self._simulated_mz_tolerance is not None
                and self._simulated_intensity_threshold is not None
            ):
                if self._isotope_envelope_backend is not None:
                    predicted_envelope = (
                        self._isotope_envelope_backend.simulate_isotope_envelope(
                            row_list,
                            self._adduct_elements,
                            self._ion_charge,
                            self._simulated_mz_tolerance,
                            self._simulated_intensity_threshold,
                        )
                    )
                else:
                    from ..isotopes.envelope import get_isotope_envelope
                    ion_formula = self._build_ion_formula(
                        idx=idx,
                        row_list=row_list,
                        core_formula=formula,
                    )
                    if ion_formula is not None:
                        predicted_envelope = get_isotope_envelope(
                            formula=ion_formula,
                            mz_tolerance=self._simulated_mz_tolerance,
                            threshold=self._simulated_intensity_threshold,
                        )

            rmse, match_frac, n_matched, peak_matches = isotope_row
            isotope_result = IsotopeMatchResult(
                num_peaks_matched=n_matched,
                num_peaks_total=self._n_obs,
                intensity_rmse=rmse,
                match_fraction=match_frac,
                peak_matches=peak_matches,
                predicted_envelope=predicted_envelope,
            )

        candidate = FormulaCandidate(
            formula=formula,
            error_ppm=error_ppm,
            error_da=error_da,
            rdbe=rdbe,
            adduct=self._adduct,
            isotope_match_result=isotope_result,
            prior_score=prior_score,
            posterior_score=posterior_score,
        )
        self._cache[idx] = candidate
        return candidate

    def _reindex(self, idx) -> '_LazyBackend':
        """
        Return a new _LazyBackend reindexed
        by slice, boolean mask, or int array
        """
        if self._rust_result is not None:
            indices = self._indices_from_selector(idx)
            return self._with_rust_result(self._rust_result.take_indices(indices))

        raw = {
            'counts': self._counts[idx],
            'exact_masses': self._exact_masses[idx],
            'error_ppm': self._error_ppm[idx],
            'error_da': self._error_da[idx],
        }
        if self._rdbe is not None:
            raw['rdbe'] = self._rdbe[idx]
        if self._iso_rmse is not None:
            raw['iso_rmse'] = self._iso_rmse[idx]
            raw['iso_match_frac'] = self._iso_match_frac[idx]
            raw['iso_n_matched'] = self._iso_n_matched[idx]
        if self._iso_peak_matches is not None:
            raw['iso_peak_matches'] = self._iso_peak_matches[idx]
        if self._formula_strings is not None:
            if isinstance(idx, (int, np.integer)):
                raw['formula_strings'] = [self._formula_strings[int(idx)]]
            else:
                raw['formula_strings'] = list(np.asarray(self._formula_strings, dtype=object)[idx])
        return _LazyBackend(
            raw=raw,
            symbols=self._symbols,
            charge=self._charge,
            ion_charge=self._ion_charge,
            adduct=self._adduct,
            adduct_elements=self._adduct_elements,
            n_obs=self._n_obs,
            charge_mass_offset=self._charge_mass_offset,
            adduct_mass=self._adduct_mass,
            simulated_mz_tolerance=self._simulated_mz_tolerance,
            simulated_intensity_threshold=self._simulated_intensity_threshold,
            isotope_envelope_backend=self._isotope_envelope_backend,
        )

    def _indices_from_selector(self, idx) -> list[int]:
        if isinstance(idx, (int, np.integer)):
            value = int(idx)
            if value < 0:
                value += len(self)
            return [value]
        if isinstance(idx, slice):
            return list(range(len(self)))[idx]

        arr = np.asarray(idx)
        if arr.dtype == bool:
            return np.flatnonzero(arr).astype(np.intp).tolist()

        values = arr.astype(np.intp, copy=False).tolist()
        if isinstance(values, int):
            values = [values]
        return [value + len(self) if value < 0 else value for value in values]

    def _with_rust_result(self, rust_result) -> '_LazyBackend':
        return _LazyBackend(
            raw={'rust_result': rust_result},
            symbols=self._symbols,
            charge=self._charge,
            ion_charge=self._ion_charge,
            adduct=self._adduct,
            adduct_elements=self._adduct_elements,
            n_obs=self._n_obs,
            charge_mass_offset=self._charge_mass_offset,
            adduct_mass=self._adduct_mass,
            simulated_mz_tolerance=self._simulated_mz_tolerance,
            simulated_intensity_threshold=self._simulated_intensity_threshold,
            isotope_envelope_backend=self._isotope_envelope_backend,
        )

    def _sort_by_error(self, reverse: bool = False) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(self._rust_result.sort_by_error(reverse))

        order = np.argsort(np.abs(self._error_da))
        if reverse:
            order = order[::-1]
        return self._reindex(order)

    def _sort_by_rmse(self, reverse: bool = False) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(self._rust_result.sort_by_rmse(reverse))

        if self._iso_rmse is None:
            return self
        order = np.argsort(self._iso_rmse)
        if reverse:
            order = order[::-1]
        return self._reindex(order)

    def _sort_by_prior(self, reverse: bool = False) -> '_LazyBackend | None':
        if self._rust_result is None or self._cache:
            return None
        return self._with_rust_result(self._rust_result.sort_by_prior(reverse))

    def _sort_by_posterior(self, reverse: bool = False) -> '_LazyBackend | None':
        if self._rust_result is None or self._cache:
            return None
        return self._with_rust_result(
            self._rust_result.sort_by_posterior(reverse)
        )

    def _score_with_prior(
        self,
        ratio_elements: list[str],
        p_absent: list[float],
        kde_points: list[list[float]],
        kde_weights: list[list[float]],
        kde_variance: list[float],
        uniform_weight: float,
        mass_sigma_ppm: float,
        isotope_sigma: float,
    ) -> bool:
        if self._rust_result is None or self._cache:
            return False
        self._rust_result = self._rust_result.score_prior(
            self._symbols,
            ratio_elements,
            p_absent,
            kde_points,
            kde_weights,
            kde_variance,
            uniform_weight,
            mass_sigma_ppm,
            isotope_sigma,
        )
        self._formula_strings = None
        return True

    def _filter_by_rdbe_range(
        self,
        min_rdbe: float,
        max_rdbe: float,
    ) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(
                self._rust_result.filter_by_rdbe(min_rdbe, max_rdbe)
            )

        if self._rdbe is None:
            return self._reindex(np.zeros(len(self), dtype=bool))
        mask = (self._rdbe >= min_rdbe) & (self._rdbe <= max_rdbe)
        return self._filter_by_mask(mask)

    def _filter_by_error_limits(
        self,
        max_ppm: float | None = None,
        max_da: float | None = None,
    ) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(
                self._rust_result.filter_by_error(max_ppm, max_da)
            )

        mask = np.ones(len(self), dtype=bool)
        if max_ppm is not None:
            mask &= np.abs(self._error_ppm) <= max_ppm
        if max_da is not None:
            mask &= np.abs(self._error_da) <= max_da
        return self._filter_by_mask(mask)

    def _filter_by_isotope_quality_limits(
        self,
        max_match_rmse: float,
        min_match_fraction: float,
    ) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(
                self._rust_result.filter_by_isotope_quality(
                    max_match_rmse,
                    min_match_fraction,
                )
            )

        if self._iso_rmse is None:
            return self._reindex(np.zeros(len(self), dtype=bool))
        mask = (
            (self._iso_rmse <= max_match_rmse)
            & (self._iso_match_frac >= min_match_fraction)
        )
        return self._filter_by_mask(mask)

    def _filter_by_octet_rule(self) -> '_LazyBackend':
        if self._rust_result is not None:
            return self._with_rust_result(
                self._rust_result.filter_by_octet(self._charge)
            )

        filtered_indices = [
            idx for idx in range(len(self))
            if passes_octet_rule(self._materialize(idx).formula)
        ]
        return self._reindex(np.asarray(filtered_indices, dtype=np.intp))

    def _slice(self, s: slice) -> '_LazyBackend':
        """
        Return a new _LazyBackend for a slice of the data
        """
        return self._reindex(s)

    def _filter_by_mask(self, mask: np.ndarray) -> '_LazyBackend':
        """
        Return a new _LazyBackend filtered by boolean mask
        """
        return self._reindex(mask)

    def _render_table(self, max_rows: Optional[int]) -> str | None:
        if self._rust_result is None or self._cache:
            return None
        rust_max_rows = None if max_rows is None else int(max_rows)
        return _render_raw_table(
            self._rust_result.table_rows(rust_max_rows),
            max_rows=max_rows,
            total=len(self),
        )

    def _to_dataframe(self) -> 'pd.DataFrame | None':
        if self._rust_result is None or self._cache:
            return None
        return _render_raw_dataframe(self._rust_result.table_rows(None))


@dataclass
class FormulaSearchResults:
    """
    Container for formula search results with filtering and display methods

    This class wraps a list of FormulaCandidate objects and provides:
    - Iterator/indexing support for easy access to MF candidates
    - Post-hoc filtering methods that return new FormulaSearchResults
    - Formatted representation in response to `print()`
    - Formatted table output via to_table()
    - Optional pandas DataFrame export

    Attributes:
        candidates: List of formula candidates
        query_mass: The mass that was searched
        query_params: Dictionary of search parameters used

    Example:
        >>> finder: 'FormulaFinder'
        >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
        >>> print(results)  # Gives a summary
        >>> for candidate in results:  # Iterate
        ...     print(candidate.formula)
        >>> # Post-hoc filter:
        >>> filtered: FormulaSearchResults = results.filter_by_rdbe(0, 10)
    """
    candidates: list[FormulaCandidate]
    query_mass: float
    query_params: dict = field(default_factory=dict)
    _backend: _LazyBackend | None = field(default=None, repr=False)

    def __getattribute__(self, name):
        if name == 'candidates':
            backend = object.__getattribute__(self, '_backend')
            if backend is not None:
                return [
                    backend._materialize(i) for i in range(len(backend))
                ]
        return super().__getattribute__(name)

    def __len__(self) -> int:
        if self._backend is not None:
            return len(self._backend)
        return len(self.candidates)

    def __iter__(self):
        if self._backend is not None:
            return (self._backend._materialize(i) for i in range(len(self._backend)))
        return iter(self.candidates)

    @overload
    def __getitem__(
        self,
        idx: int,
    ) -> FormulaCandidate: ...

    @overload
    def __getitem__(
        self,
        idx: slice
    ) -> 'FormulaSearchResults': ...

    def __getitem__(
        self,
        idx: int | slice,
    ) -> 'FormulaCandidate | FormulaSearchResults':
        """
        Either returns a formula candidate, or a new
        FormulaSearchResults instance if given a slice

        Args:
            idx: index or slice of list

        Returns:
            Either a FormulaCandidate object, or
            another FormulaSearchResults instance
        """
        if self._backend is not None:
            if isinstance(idx, slice):
                return FormulaSearchResults(
                    candidates=[],
                    query_mass=self.query_mass,
                    query_params=self.query_params,
                    _backend=self._backend._slice(idx),
                )
            if idx < 0:
                idx += len(self._backend)
            return self._backend._materialize(idx)

        if isinstance(idx, slice):
            return FormulaSearchResults(
                candidates=self.candidates[idx],
                query_mass=self.query_mass,
                query_params=self.query_params,
            )

        return self.candidates[idx]

    def __repr__(self) -> str:
        """
        Text summary and top candidates
        """
        n_results = len(self)
        summary = self._summary_line(n_results)

        if n_results == 0:
            return summary

        # Show top 5 candidates
        lines = [summary, "", self.to_table(max_rows=5)]
        return "\n".join(lines)

    # === FORMATTING METHODS ===
    def _summary_line(self, n_results: int) -> str:
        """
        Build the header line, including adduct notation when present.
        """
        adduct = self.query_params.get('adduct')
        charge = self.query_params.get('charge', 0)
        parts = [
            f"query_mass={self.query_mass:.4f}",
            f"n_results={n_results}",
        ]

        if adduct is not None:
            adduct_part = adduct if adduct.startswith('-') else f'+{adduct}'
            sign = '+' if charge > 0 else '-' if charge < 0 else ''
            abs_charge = abs(charge)
            charge_str = f'{abs_charge}{sign}' if abs_charge > 1 else sign
            parts.append(f"adduct=[M{adduct_part}]{charge_str}")

        return f"FormulaSearchResults({', '.join(parts)})"

    def to_table(
        self,
        max_rows: Optional[int] = None
    ) -> str:
        """
        Return formatted table of all candidates

        Args:
            max_rows: Maximum number of rows to display. None shows all.

        Returns:
            Formatted string table
        """
        if self._backend is not None:
            table = self._backend._render_table(max_rows)
            if table is not None:
                return table

        n_results = len(self)
        if max_rows is None:
            indices = range(n_results)
        else:
            indices = list(range(n_results))[:max_rows]
        candidates_to_show = [self[idx] for idx in indices]

        return render_table(
            candidates_to_show,
            max_rows=max_rows,
            total=n_results,
        )
        #
        # n = len(self)
        # if n == 0:
        #     return "No candidates found."
        #
        # show_n = n if max_rows is None else min(n, max_rows)
        #
        # # Materialize only the rows we need to display
        # candidates_to_show = [self[i] for i in range(show_n)]
        #
        # # Check if any candidates have isotope/fragment matching results
        # has_isotope_results = any(
        #     c.isotope_match_result is not None for c in candidates_to_show
        # )
        # # Build header dynamically
        # header = f"{'Formula':<25} {'Error (ppm)':<15} {'Error (Da)':<15} {'RDBE':<10}"
        # sep_len = 70
        #
        # if has_isotope_results:
        #     header += f" {'Iso. Matches':<15}"
        #     header += f"{'Iso. RMSE':<10}"
        #     sep_len += 26
        #
        # lines: list[str] = [header, "-" * sep_len]
        #
        # # Build rows
        # for candidate in candidates_to_show:
        #     formula_str = candidate.formula.formula
        #     rdbe_str = f"{candidate.rdbe:.1f}" if candidate.rdbe is not None else "N/A"
        #
        #     iso_match_str = ""
        #     iso_score_str = ""
        #     if candidate.isotope_match_result is not None:
        #         iso_match_str = (f"{candidate.isotope_match_result.num_peaks_matched}"
        #                          f"/{candidate.isotope_match_result.num_peaks_total}")
        #         iso_score_str = f"{candidate.isotope_match_result.intensity_rmse:.4f}"
        #
        #     if has_isotope_results:
        #         lines.append(
        #             f"{formula_str:<25} {candidate.error_ppm:>14.2f} "
        #             f"{candidate.error_da:>14.6f} {rdbe_str:>9} {iso_match_str:>13} {iso_score_str:>9}"
        #         )
        #     else:
        #         lines.append(
        #             f"{formula_str:<25} {candidate.error_ppm:>14.2f} "
        #             f"{candidate.error_da:>14.6f} {rdbe_str:>9}"
        #         )
        #
        # if max_rows is not None and n > max_rows:
        #     lines.append(f"... and {n - max_rows} more")
        #
        # return "\n".join(lines)

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert results to pandas DataFrame, if pandas is installed.

        Columns match those shown in to_table(), with conditional columns
        (isotope scores, prior score) included only when present.

        Returns:
            pandas.DataFrame with columns for formula, errors, RDBE, and
            any scored columns that were computed

        Raises:
            ImportError: If pandas is not installed
        """
        if self._backend is not None:
            dataframe = self._backend._to_dataframe()
            if dataframe is not None:
                return dataframe
        return render_dataframe(list(self))
        # try:
        #     import pandas as pd
        # except ImportError:
        #     raise ImportError(
        #         "pandas is required for to_dataframe(). "
        #         "Install with: pip install pandas"
        #     )
        #
        # # Fast path: read directly from backend arrays
        # if self._backend is not None:
        #     b = self._backend
        #     n = len(b)
        #     data = {
        #         'formula': [
        #             b._materialize(i).formula.formula for i in range(n)
        #         ],
        #         'error_ppm': b._error_ppm.tolist(),
        #         'error_da': b._error_da.tolist(),
        #         'rdbe': b._rdbe.tolist() if b._rdbe is not None else [None] * n,
        #         'mass': b._exact_masses.tolist(),
        #     }
        #     if b._iso_rmse is not None:
        #         data['isotope_intensity_rmse'] = b._iso_rmse.tolist()
        #         data['isotope_match_fraction'] = b._iso_match_frac.tolist()
        #     return pd.DataFrame(data)
        #
        # data = []
        # for candidate in self.candidates:
        #     row = {
        #         'formula': candidate.formula.formula,
        #         'error_ppm': candidate.error_ppm,
        #         'error_da': candidate.error_da,
        #         'rdbe': candidate.rdbe,
        #         'mass': candidate.formula.monoisotopic_mass,
        #     }
        #
        #     if candidate.isotope_match_result is not None:
        #         if isinstance(
        #             candidate.isotope_match_result, IsotopeMatchResult
        #         ):
        #             row['isotope_intensity_rmse'] = candidate.isotope_match_result.intensity_rmse
        #             row['isotope_match_fraction'] = candidate.isotope_match_result.match_fraction
        #
        #     data.append(row)
        #
        # return pd.DataFrame(data)

    # === SORTING METHODS ===
    def sort_by_error(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by absolute mass error (Da).

        Args:
            reverse: If True, sort in descending order (largest error first)

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            return FormulaSearchResults(
                candidates=[],
                query_mass=self.query_mass,
                query_params=self.query_params,
                _backend=self._backend._sort_by_error(reverse),
            )

        return FormulaSearchResults(
            candidates=sorted(self.candidates, reverse=reverse),
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def sort_by_rmse(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by isotope intensity RMSE.

        Candidates without isotope match results are placed at the end.

        Args:
            reverse: If True, sort in descending order (largest RMSE first)

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            return FormulaSearchResults(
                candidates=[],
                query_mass=self.query_mass,
                query_params=self.query_params,
                _backend=self._backend._sort_by_rmse(reverse),
            )

        with_iso = [c for c in self.candidates if c.isotope_match_result is not None]
        without_iso = [c for c in self.candidates if c.isotope_match_result is None]

        sorted_with = sorted(
            with_iso,
            key=lambda x: x.isotope_match_result.intensity_rmse,
            reverse=reverse,
        )

        return FormulaSearchResults(
            candidates=sorted_with + without_iso,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def sort_by_prior(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by prior score.

        Candidates without prior scores are placed at the end.

        Args:
            reverse: If True, sort in ascending order (lowest score first).
                By default, sorts descending (highest/most plausible first).

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            new_backend = self._backend._sort_by_prior(reverse)
            if new_backend is not None:
                return FormulaSearchResults(
                    candidates=[],
                    query_mass=self.query_mass,
                    query_params=self.query_params,
                    _backend=new_backend,
                )

        with_score = [c for c in self.candidates if c.prior_score is not None]
        without_score = [c for c in self.candidates if c.prior_score is None]

        sorted_with = sorted(
            with_score,
            key=lambda x: x.prior_score,
            reverse=not reverse,
        )

        return FormulaSearchResults(
            candidates=sorted_with + without_score,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    def sort_by_posterior(
        self,
        reverse: bool = False,
    ) -> 'FormulaSearchResults':
        """
        Sort candidates by posterior score.

        Candidates without posterior scores are placed at the end.

        Args:
            reverse: If True, sort in ascending order (lowest score first).
                By default, sorts descending (highest/most plausible first).

        Returns:
            New FormulaSearchResults with sorted candidates
        """
        if self._backend is not None:
            new_backend = self._backend._sort_by_posterior(reverse)
            if new_backend is not None:
                return FormulaSearchResults(
                    candidates=[],
                    query_mass=self.query_mass,
                    query_params=self.query_params,
                    _backend=new_backend,
                )

        with_score = [c for c in self.candidates if c.posterior_score is not None]
        without_score = [c for c in self.candidates if c.posterior_score is None]

        sorted_with = sorted(
            with_score,
            key=lambda x: x.posterior_score,
            reverse=not reverse,
        )

        return FormulaSearchResults(
            candidates=sorted_with + without_score,
            query_mass=self.query_mass,
            query_params=self.query_params,
        )

    # === FILTERING METHODS ===
    def filter_by_rdbe(
        self,
        min_rdbe: float,
        max_rdbe: float
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by RDBE range

        Args:
            min_rdbe: Minimum RDBE value (inclusive)
            max_rdbe: Maximum RDBE value (inclusive)

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        if self._backend is not None:
            b = self._backend
            new_backend = b._filter_by_rdbe_range(min_rdbe, max_rdbe)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
                _backend=new_backend,
            )

        filtered = [
            c for c in self.candidates
            if c.rdbe is not None and min_rdbe <= c.rdbe <= max_rdbe
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'filter_rdbe': (min_rdbe, max_rdbe),
            }
        )

    def filter_by_octet(self) -> 'FormulaSearchResults':
        """
        Filter candidates to only those passing the octet rule.

        Returns:
            New FormulaSearchResults with filtered candidates
        """
        if self._backend is not None:
            return FormulaSearchResults(
                candidates=[],
                query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'check_octet': True,
                },
                _backend=self._backend._filter_by_octet_rule(),
            )

        filtered = [
            c for c in self
            if passes_octet_rule(c.formula)
        ]

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'check_octet': True,
            }
        )

    def filter_by_error(
        self,
        max_ppm: Optional[float] = None,
        max_da: Optional[float] = None
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by maximum error.

        At least one of max_ppm or max_da must be specified.

        Args:
            max_ppm: Maximum absolute error in ppm
            max_da: Maximum absolute error in Da

        Returns:
            New FormulaSearchResults with filtered candidates

        Raises:
            ValueError: If neither max_ppm nor max_da is specified
        """
        if max_ppm is None and max_da is None:
            raise ValueError(
                "At least one of max_ppm or max_da must be specified"
            )

        if self._backend is not None:
            b = self._backend
            new_backend = b._filter_by_error_limits(max_ppm, max_da)
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'max_error_ppm': max_ppm, 'max_error_da': max_da,
                },
                _backend=new_backend,
            )

        filtered = []
        for c in self.candidates:
            passes = True
            if max_ppm is not None and abs(c.error_ppm) > max_ppm:
                passes = False
            if max_da is not None and abs(c.error_da) > max_da:
                passes = False
            if passes:
                filtered.append(c)

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'max_error_ppm': max_ppm,
                'max_error_da': max_da
            }
        )

    def filter_by_isotope_quality(
        self,
        max_match_rmse: Optional[float] = 1.0,
        min_match_fraction: Optional[float] = 0.0,
    ) -> 'FormulaSearchResults':
        """
        Filter candidates by isotope match quality.

        Uses isotope matching results to filter candidate formulae.

        Args:
            max_match_rmse: Maximum isotope envelope RMSE.
                Example: 0.05 means the total error in isotope envelope can't
                exceed 5%.
                Default: 1.0 (total error can't exceed 100%)

            min_match_fraction: Minimum fraction of peaks matched (0.0-1.0)
                Example: 0.8 means at least 80% of peaks must match
                Default: 0.0 (no filter)

        Returns:
            New FormulaSearchResults with filtered candidates

        Raises:
            ValueError: If neither parameter is specified or if candidates
                don't have isotope match results
        """
        if self._backend is not None:
            b = self._backend
            new_backend = b._filter_by_isotope_quality_limits(
                max_match_rmse,
                min_match_fraction,
            )
            return FormulaSearchResults(
                candidates=[], query_mass=self.query_mass,
                query_params={
                    **self.query_params,
                    'min_match_fraction': min_match_fraction,
                    'max_match_rmse': max_match_rmse,
                },
                _backend=new_backend,
            )

        filtered = []
        for c in self.candidates:
            if c.isotope_match_result is None:
                continue
            if c.isotope_match_result.match_fraction < min_match_fraction:
                continue
            if c.isotope_match_result.intensity_rmse > max_match_rmse:
                continue
            filtered.append(c)

        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={
                **self.query_params,
                'min_match_fraction': min_match_fraction,
                'max_match_rmse': max_match_rmse,
            }
        )

    def get_isotope_details(
        self,
        index: int
    ) -> 'IsotopeMatchResult | None':
        """
        Get detailed isotope matching information for a specific MF candidate.

        Args:
            index: Index of the candidate to inspect

        Returns:
            IsotopeMatchResult (SingleEnvelopeMatchResult)
            with detailed per-peak information, or None if no isotope matching
            was performed for this candidate

        Example:
            >>> finder: 'FormulaFinder'
            >>> results = finder.find_formulae(...)
            >>> details = results.get_isotope_details(0)
            >>> if details:
            ...     print(f"Matched {details.num_peaks_matched}/{details.num_peaks_total}")
            ...     print(f"Per-peak: {details.peak_matches}")
        """
        n = len(self)
        if index < 0 or index >= n:
            raise IndexError(
                f"Index {index} out of range for {n} candidates"
            )

        return self[index].isotope_match_result

    def top(
        self,
        n: int = 10,
    ) -> 'FormulaSearchResults':
        """
        Return top N candidates by error.

        Args:
            n: Number of top candidates to return

        Returns:
            New FormulaSearchResults with top N candidates
        """
        return self[:n]
