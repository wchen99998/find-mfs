"""
Self-contained molecular formula finder from accurate mass values.

Implements Böcker & Lipták's algorithm for mass decomposition using Extended
Residue Tables (ERT), with RDBE/octet chemical validation.

Dependencies: numpy, numba, molmass

Usage:
    >>> finder = FormulaFinder('CHNOPS')
    >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
    >>> print(results)

    >>> # With filters
    >>> results = finder.find_formulae(
    ...     mass=500.0, error_ppm=5.0,
    ...     filter_rdbe=(0, 20), check_octet=True,
    ... )

    >>> # With adduct
    >>> results = finder.find_formulae(
    ...     mass=203.053, charge=1, adduct='Na', error_ppm=5.0,
    ... )
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from typing import Optional, Union, Iterable, overload

import numpy as np
from molmass import Formula, Composition, CompositionItem
from molmass.elements import ELEMENTS, ELECTRON
from numba import njit, int32, float64, types
from numba.typed import List as NumbaList
from numba.experimental import jitclass


# ============================================================================
# Constants
# ============================================================================

BOND_ELECTRONS: dict[str, int] = {
    'H': 1, 'Li': 1, 'Na': 1, 'C': 4, 'N': 3, 'O': 2,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'S': 2, 'P': 5,
    'Si': 4, 'B': 3,
}


# ============================================================================
# Utility functions (from utils/filtering.py and utils/formulae.py)
# ============================================================================

def _is_half_integer(x: float) -> bool:
    return (2 * x) % 2 == 1


def get_bond_electrons(symbol: str) -> Optional[int]:
    return BOND_ELECTRONS.get(symbol, None)


def get_rdbe(formula: Union[Formula, 'LightFormula']) -> Optional[float]:
    """
    Calculate Ring and Double Bond Equivalents (RDBE).

    RDBE = 0.5 * sum(n_i * (b_i - 2)) + 1
    """
    n_b_sub_2: list[int] = []
    for element in formula.composition().values():
        if element.symbol == 'e-':
            continue
        count = element.count
        num_bond_eles = get_bond_electrons(element.symbol)
        if not num_bond_eles:
            return None
        n_b_sub_2.append(count * (num_bond_eles - 2))
    return (0.5 * sum(n_b_sub_2)) + 1


def passes_octet_rule(formula: Union[Formula, 'LightFormula']) -> bool:
    """Check if a molecular formula satisfies the octet rule."""
    rdbe = get_rdbe(formula)
    if rdbe is None:
        return False
    charge = formula.charge
    if abs(charge) % 2.0 == 0.0:
        return not _is_half_integer(rdbe)
    elif abs(charge) % 2.0 == 1.0:
        return _is_half_integer(rdbe)
    else:
        raise ValueError(f"Invalid charge: {charge}")


def to_bounds_dict(
    formula: str,
    elements: Iterable[str],
) -> dict[str, int]:
    """
    Convert constraint strings like "C20H10O5P0" into element bounds dicts.

    Elements mentioned get their specified counts. Unmentioned elements
    default to 0. "*" means no bound (infinity). No number means 1.
    """
    pattern = r'([A-Z][a-z]?)(\d+|\*)?'
    matches = re.findall(pattern, formula)

    parsed = {}
    for symbol, count in matches:
        if not symbol:
            continue
        if symbol not in ELEMENTS:
            raise ValueError(f"Invalid element symbol: '{symbol}'")
        if symbol not in elements:
            raise ValueError(
                f"Element '{symbol}' is not in the given element set: {elements}"
            )
        if count == "*":
            parsed[symbol] = float('inf')
        elif count:
            parsed[symbol] = int(count)
        else:
            parsed[symbol] = 1

    output = {k: 0 for k in elements}
    output.update(parsed)
    return output


# ============================================================================
# LightFormula (from core/light_formula.py)
# ============================================================================

class LightFormula:
    """
    Lightweight formula class that avoids expensive molmass.Formula parsing.

    When element symbols, counts, charge, and monoisotopic mass are already
    known, constructing a full molmass.Formula is wasteful. LightFormula
    stores these pre-computed values directly.
    """

    __slots__ = (
        '_elements', '_symbols', '_counts',
        '_charge', '_monoisotopic_mass', '_formula_str',
    )

    def __init__(
        self,
        elements: dict[str, int] | None = None,
        charge: int = 0,
        monoisotopic_mass: float = 0.0,
        symbols: list[str] | tuple[str, ...] | None = None,
        counts: list[int] | tuple[int, ...] | None = None,
    ):
        if elements is not None and (symbols is not None or counts is not None):
            raise ValueError(
                "Provide either `elements` or (`symbols`, `counts`), not both."
            )
        if elements is None:
            if symbols is None and counts is None:
                elements = {}
            elif symbols is None or counts is None:
                raise ValueError(
                    "When `elements` is None, both `symbols` and `counts` are required."
                )
            elif len(symbols) != len(counts):
                raise ValueError(
                    f"`symbols` and `counts` must have same length; got "
                    f"{len(symbols)} and {len(counts)}"
                )
        self._elements = elements
        self._symbols = symbols
        self._counts = counts
        self._charge = charge
        self._monoisotopic_mass = monoisotopic_mass
        self._formula_str: str | None = None

    @classmethod
    def from_counts(
        cls,
        symbols: list[str] | tuple[str, ...],
        counts: list[int] | tuple[int, ...],
        charge: int = 0,
        monoisotopic_mass: float = 0.0,
    ) -> 'LightFormula':
        """Construct from parallel symbols/counts with minimal overhead."""
        obj = cls.__new__(cls)
        obj._elements = None
        obj._symbols = symbols
        obj._counts = counts
        obj._charge = charge
        obj._monoisotopic_mass = monoisotopic_mass
        obj._formula_str = None
        return obj

    @property
    def formula(self) -> str:
        """Hill-notation formula string with charge notation."""
        if self._formula_str is None:
            self._formula_str = self._build_formula_str()
        return self._formula_str

    @property
    def monoisotopic_mass(self) -> float:
        return self._monoisotopic_mass

    @property
    def charge(self) -> int:
        return self._charge

    @property
    def empirical(self) -> str:
        nonzero = {}
        for symbol, count in self._iter_nonzero_items():
            nonzero[symbol] = count
        if not nonzero:
            return ''
        g = reduce(gcd, nonzero.values())
        reduced = {s: c // g for s, c in nonzero.items()}
        parts: list[str] = []
        sorted_symbols = sorted(
            reduced,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )
        for sym in sorted_symbols:
            cnt = reduced[sym]
            parts.append(sym if cnt == 1 else f'{sym}{cnt}')
        return ''.join(parts)

    @property
    def atoms(self) -> int:
        total = 0
        for _, count in self._iter_nonzero_items():
            total += count
        return total

    def composition(self) -> Composition:
        """Build a molmass.Composition matching Formula.composition() output."""
        nonzero_items: list[tuple[str, int]] = []
        total_mass = 0.0
        for symbol, count in self._iter_nonzero_items():
            nonzero_items.append((symbol, count))
            total_mass += ELEMENTS[symbol].mass * count

        items: list[tuple[str, int, float, float]] = []
        for symbol, count in nonzero_items:
            mass = ELEMENTS[symbol].mass * count
            fraction = mass / total_mass if total_mass > 0 else 0.0
            items.append((symbol, count, mass, fraction))

        if self._charge != 0:
            items.append(('e-', -self._charge, 0.0, 0.0))

        return Composition(items)

    def to_formula(self) -> Formula:
        return Formula(self.formula)

    def __add__(self, other):
        if isinstance(other, LightFormula):
            merged = {sym: cnt for sym, cnt in self._iter_nonzero_items()}
            for sym, cnt in other._iter_nonzero_items():
                merged[sym] = merged.get(sym, 0) + cnt
            return LightFormula(
                elements=merged,
                charge=self._charge + other._charge,
                monoisotopic_mass=self._monoisotopic_mass + other._monoisotopic_mass,
            )
        if isinstance(other, Formula):
            merged = {sym: cnt for sym, cnt in self._iter_nonzero_items()}
            for sym, item in other.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                merged[sym] = merged.get(sym, 0) + item.count
            return LightFormula(
                elements=merged,
                charge=self._charge + other.charge,
                monoisotopic_mass=self._monoisotopic_mass + other.monoisotopic_mass,
            )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Formula):
            return self.__add__(other)
        return NotImplemented

    def __str__(self) -> str:
        return self.formula

    def __repr__(self) -> str:
        return f"LightFormula('{self.formula}')"

    def _build_formula_str(self) -> str:
        parts: list[str] = []
        nonzero = {}
        for symbol, count in self._iter_nonzero_items():
            nonzero[symbol] = count
        sorted_symbols = sorted(
            nonzero,
            key=lambda s: (0,) if s == 'C' else (1,) if s == 'H' else (2, s),
        )
        for sym in sorted_symbols:
            cnt = nonzero[sym]
            if cnt == 1:
                parts.append(sym)
            else:
                parts.append(f'{sym}{cnt}')
        base = ''.join(parts)
        if self._charge == 0:
            return base
        sign = '+' if self._charge > 0 else '-'
        abs_charge = abs(self._charge)
        if abs_charge == 1:
            return f'[{base}]{sign}'
        return f'[{base}]{abs_charge}{sign}'

    def _iter_nonzero_items(self):
        if self._elements is not None:
            for symbol, count in self._elements.items():
                if count > 0:
                    yield symbol, count
            return
        symbols = self._symbols
        counts = self._counts
        if symbols is None or counts is None:
            return
        for i in range(len(symbols)):
            count = counts[i]
            if count > 0:
                yield symbols[i], count


# ============================================================================
# Numba JIT algorithms (from core/algorithms.py)
# ============================================================================

@njit(cache=True)
def _gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return a


@njit(cache=True, inline='always')
def _is_decomposable(
        ERT: np.ndarray,
        i: int,
        m: int,
        a1: int,
) -> bool:
    """Check if mass m is decomposable using first i+1 elements."""
    if m < 0:
        return False
    if ERT[int(m % a1), i] <= m:
        return True
    else:
        return False


@njit(cache=True, fastmath=True)
def _decompose_mass_range(
        ERT: np.ndarray,
        integer_masses: np.ndarray,
        real_masses: np.ndarray,
        bounds: np.ndarray,
        min_values: np.ndarray,
        min_int: int,
        max_int: int,
        original_min_mass: float,
        original_max_mass: float,
        charge_mass_offset: float,
        max_results: int,
        rdbe_coeffs: np.ndarray,
        rdbe_min: float,
        rdbe_max: float,
        check_octet: bool,
        charge_parity_even: bool,
        do_rdbe_filter: bool,
) -> np.ndarray:
    """
    Consolidated decomposition across an integer mass range.

    Iterates over all integer masses in [min_int, max_int], performs
    decomposition with bounds checking, applies min_values and exact
    mass filtering internally, and returns valid element count vectors.

    Uses fastmath=True to enable FMA fusion in the exact mass dot product.
    Pre-allocates the output buffer at max_results to avoid dynamic growth.

    When do_rdbe_filter is True, applies RDBE range and octet rule checks
    at the leaf level, avoiding writing filtered-out candidates to the
    result array entirely.

    Returns:
        2D int32 array of shape (N, num_elements) with valid counts
    """
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1

    result = np.empty((max_results, num_elements), dtype=np.int32)
    count = 0
    # Track total mass-valid candidates (before RDBE filter) to preserve
    # the same stopping semantics: stop after max_results candidates pass
    # the mass filter, regardless of RDBE outcome.
    mass_valid_count = 0

    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and mass_valid_count < max_results:
            if not _is_decomposable(ERT, i, m, a1):
                while i <= k and not _is_decomposable(ERT, i, m, a1):
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1
            else:
                while i > 0 and _is_decomposable(ERT, i - 1, m, a1):
                    i -= 1

                if i == 0:
                    c[0] = m // a1

                    if c[0] <= bounds[0]:
                        # Compute exact mass and RDBE in a single fused loop
                        total = np.int64(0)
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        for j in range(num_elements):
                            val = c[j] + min_values[j]
                            val_f = np.float64(val)
                            total += val
                            exact_mass += val_f * real_masses[j]
                            rdbe += val_f * rdbe_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
                            mass_valid_count += 1
                            store = True
                            if do_rdbe_filter:
                                if rdbe < rdbe_min or rdbe > rdbe_max:
                                    store = False
                                elif check_octet:
                                    doubled_int = np.int64(2.0 * rdbe)
                                    is_half_int = (doubled_int & 1) == 1
                                    if charge_parity_even and is_half_int:
                                        store = False
                                    elif not charge_parity_even and not is_half_int:
                                        store = False

                            if store:
                                for j in range(num_elements):
                                    result[count, j] = np.int32(c[j] + min_values[j])
                                count += 1

                    i += 1

                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return result[:count]


# ============================================================================
# Element jitclass and MassDecomposer (from core/decomposer.py)
# ============================================================================

_element_spec = [
    ('symbol', types.unicode_type),
    ('mass', float64),
    ('integer_mass', int32),
]


@jitclass(_element_spec)
class Element:
    """Numba-compatible element with mass and symbol."""
    def __init__(self, symbol: str, mass: float):
        self.symbol = symbol
        self.mass = mass
        self.integer_mass = 0


def _get_element_most_abundant_mass(element) -> float:
    """Return mass of the most abundant isotope."""
    isotopes = sorted(
        element.isotopes.values(),
        key=lambda x: x.abundance,
        reverse=True,
    )
    return isotopes[0].mass


def _append_charge(formula_str: str, charge: int) -> str:
    if charge == 0:
        return formula_str
    sign = "+" if charge > 0 else "-"
    return formula_str + sign * abs(charge)


class MassDecomposer:
    """
    Decomposes a mass into candidate molecular formulae using the
    Böcker & Lipták Extended Residue Table algorithm.
    """

    def __init__(
        self,
        elements: Iterable[str] = 'CHNOPS',
        use_precalculated: bool = True,
    ):
        if isinstance(elements, str):
            elements = list(Formula(elements).composition().keys())

        if use_precalculated and self._try_load_precalculated(elements):
            return

        self._init_from_elements(elements)

    def init_ERT(self):
        """Build extended residue table."""
        if self.ERT is not None:
            return
        self._discretize_masses()
        self._divide_by_gcd()
        self._calculate_ert()
        self._compute_errors()

    def _discretize_masses(self):
        for element in self.elements:
            element.integer_mass = int(element.mass / self.precision)

    def _divide_by_gcd(self):
        if len(self.elements) == 1:
            d = self.elements[0].integer_mass
            self.precision *= d
            self.elements[0].integer_mass = 1
        elif len(self.elements) > 1:
            d = _gcd(self.elements[0].integer_mass, self.elements[1].integer_mass)
            for i in range(2, len(self.elements)):
                d = _gcd(d, self.elements[i].integer_mass)
                if d == 1:
                    return
            self.precision *= d
            for element in self.elements:
                element.integer_mass //= d

    def _calculate_ert(self):
        first_mass = self.elements[0].integer_mass
        num_elements = len(self.elements)

        self.ERT = np.full(
            shape=(first_mass, num_elements),
            fill_value=np.inf,
            dtype=np.float64,
        )
        self.ERT[0, 0] = 0

        for i in range(1, first_mass):
            if i % first_mass == 0:
                self.ERT[i, 0] = i

        for j in range(1, num_elements):
            self.ERT[0, j] = 0
            current_mass = self.elements[j].integer_mass
            d = _gcd(first_mass, current_mass)

            for p in range(d):
                if p == 0:
                    n = 0
                else:
                    n = np.inf
                    argmin = p
                    for i in range(p, first_mass, d):
                        if self.ERT[i, j - 1] < n:
                            n = self.ERT[i, j - 1]
                            argmin = i
                    if np.isinf(n):
                        for i in range(p, first_mass, d):
                            self.ERT[i, j] = np.inf
                        continue
                    self.ERT[argmin, j] = n

                for i in range(1, first_mass // d):
                    n += current_mass
                    r = int(n % first_mass)
                    n = min(n, self.ERT[r, j - 1])
                    self.ERT[r, j] = n

    def _compute_errors(self):
        self.min_error = 0
        self.max_error = 0
        for element in self.elements:
            error = (self.precision * element.integer_mass - element.mass) / element.mass
            self.min_error = min(self.min_error, error)
            self.max_error = max(self.max_error, error)

    def _try_load_precalculated(self, elements: Iterable[str]) -> bool:
        """Try to load a pre-calculated ERT for the given element set."""
        elements = list(elements)
        known_sets = {
            frozenset(['C', 'H', 'N', 'O', 'P', 'S']): 'chnops',
            frozenset(['C', 'H', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'I', 'F']): 'chnops_halogens',
        }
        element_set = frozenset(elements)
        if element_set not in known_sets:
            return False

        ert_name = known_sets[element_set]
        try:
            if ert_name == 'chnops':
                from find_mfs.data.ert_chnops import ERT_DATA
            elif ert_name == 'chnops_halogens':
                from find_mfs.data.ert_chnops_halogens import ERT_DATA
            else:
                return False
            self._load_from_ert_data(ERT_DATA)
            return True
        except (ImportError, KeyError, AttributeError):
            return False

    def _load_from_ert_data(self, ert_data: dict):
        self.ERT = np.ascontiguousarray(ert_data['ert'], dtype=np.float64)
        self.precision = float(ert_data['precision'])
        self.min_error = float(ert_data['min_error'])
        self.max_error = float(ert_data['max_error'])
        self.element_symbols = list(ert_data['element_symbols'])

        element_masses = ert_data['element_masses']
        element_integer_masses = ert_data['element_integer_masses']

        self.elements = NumbaList()
        for i, symbol in enumerate(self.element_symbols):
            element = Element(symbol, element_masses[i])
            element.integer_mass = element_integer_masses[i]
            self.elements.append(element)

        self.real_masses = np.array(
            [e.mass for e in self.elements], dtype=np.float64,
        )
        self.integer_masses = np.array(
            [e.integer_mass for e in self.elements], dtype=np.int64,
        )

    def _init_from_elements(self, elements):
        self.elements = NumbaList()
        for symbol in elements:
            if symbol not in ELEMENTS:
                raise ValueError(f"Invalid chemical symbol: {symbol}")
            mass = _get_element_most_abundant_mass(ELEMENTS[symbol])
            self.elements.append(Element(symbol, mass))

        self.elements.sort(key=lambda e: e.mass)
        self.element_symbols = [i.symbol for i in self.elements]
        self.real_masses = np.array(
            [e.mass for e in self.elements], dtype=np.float64,
        )
        self.precision = 1.0 / 5963.337687
        self.ERT: Optional[np.ndarray] = None
        self.min_error = 0.0
        self.max_error = 0.0
        self.init_ERT()
        self.integer_masses = np.array(
            [e.integer_mass for e in self.elements], dtype=np.int64,
        )

    def decompose_to_counts(
        self,
        query_mass: float,
        charge: int = 0,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int]] = None,
        max_counts: Optional[dict[str, int]] = None,
        max_results: int = 10000,
        rdbe_coeffs: Optional[np.ndarray] = None,
        rdbe_min: float = -np.inf,
        rdbe_max: float = np.inf,
        check_octet: bool = False,
        charge_parity_even: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Decompose a mass into element count vectors.

        Returns raw count arrays instead of Formula objects, enabling
        count-based pre-filtering before expensive Formula construction.
        """
        max_counts, min_counts = self._populate_counts(max_counts, min_counts)
        bounds, min_values = self._get_bounds(max_counts, min_counts)

        adjusted_mass = query_mass + (ELECTRON.mass * charge)

        original_min_mass, original_max_mass = self._get_mass_range(
            mass=adjusted_mass, ppm_error=ppm_error, mz_error=mz_error,
        )
        min_mass, max_mass = self._get_mass_range(
            mass=adjusted_mass, ppm_error=ppm_error,
            mz_error=mz_error, min_counts=min_counts,
        )

        if max_mass > 0:
            for i, element in enumerate(self.elements):
                mass_bound = math.floor(max_mass / element.mass)
                bounds[i] = min(bounds[i], float(mass_bound))

        min_int, max_int = self._to_integer_range(
            min_mass=max(0.0, min_mass),
            max_mass=max(0.0, max_mass),
        )

        charge_mass_offset = ELECTRON.mass * charge

        integer_masses = self.integer_masses
        real_masses = self.real_masses
        bounds_arr = np.array(bounds, dtype=np.float64)
        min_values_arr = np.array(min_values, dtype=np.int64)

        do_rdbe_filter = rdbe_coeffs is not None
        if rdbe_coeffs is None:
            rdbe_coeffs_arr = np.zeros(len(self.elements), dtype=np.float64)
        else:
            rdbe_coeffs_arr = np.ascontiguousarray(rdbe_coeffs, dtype=np.float64)

        counts = _decompose_mass_range(
            ERT=self.ERT,
            integer_masses=integer_masses,
            real_masses=real_masses,
            bounds=bounds_arr,
            min_values=min_values_arr,
            min_int=min_int,
            max_int=max_int,
            original_min_mass=original_min_mass,
            original_max_mass=original_max_mass,
            charge_mass_offset=charge_mass_offset,
            max_results=max_results,
            rdbe_coeffs=rdbe_coeffs_arr,
            rdbe_min=rdbe_min,
            rdbe_max=rdbe_max,
            check_octet=check_octet,
            charge_parity_even=charge_parity_even,
            do_rdbe_filter=do_rdbe_filter,
        )

        return counts, list(self.element_symbols)

    def decompose(
        self,
        query_mass: float,
        charge: int = 0,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int]] = None,
        max_counts: Optional[dict[str, int]] = None,
        max_results: int = 10000,
    ) -> list[Formula]:
        """Decompose a mass into possible element combinations."""
        counts, symbols = self.decompose_to_counts(
            query_mass=query_mass, charge=charge,
            ppm_error=ppm_error, mz_error=mz_error,
            min_counts=min_counts, max_counts=max_counts,
            max_results=max_results,
        )
        results: list[Formula] = []
        for idx in range(len(counts)):
            row = counts[idx]
            formula_str = ''.join(
                f"{s}{c}" for s, c in zip(symbols, row) if c > 0
            )
            formula_str = _append_charge(formula_str, charge)
            results.append(Formula(formula_str))
        return results

    def _get_mass_range(
        self,
        mass: float,
        ppm_error: Optional[float] = 0.0,
        mz_error: Optional[float] = 0.0,
        min_counts: Optional[dict[str, int]] = None,
    ) -> tuple[float, float]:
        if not ppm_error and not mz_error:
            raise ValueError("Neither ppm_error nor mz_error arguments given")
        error: float = max(
            mass * (ppm_error or 0.0) / 1e6,
            mz_error or 0.0,
        )
        min_mass = mass - error
        max_mass = mass + error
        if min_counts:
            for element in self.elements:
                if element.symbol in min_counts and min_counts[element.symbol] > 0:
                    reduce_by = element.mass * min_counts[element.symbol]
                    min_mass -= reduce_by
                    max_mass -= reduce_by
        return min_mass, max_mass

    def _get_bounds(
        self,
        max_counts: Optional[dict[str, int]],
        min_counts: Optional[dict[str, int]],
    ) -> tuple[list[float], list[int]]:
        bounds: list[float] = [float('inf')] * len(self.elements)
        min_values = [0] * len(self.elements)
        for i, element in enumerate(self.elements):
            if min_counts and element.symbol in min_counts:
                min_values[i] = min_counts[element.symbol]
            if max_counts and element.symbol in max_counts:
                bounds[i] = float(max_counts[element.symbol] - min_values[i])
        return bounds, min_values

    def _to_integer_range(
        self,
        min_mass: float,
        max_mass: float,
    ) -> tuple[int, int]:
        from_int = math.ceil((1 + self.min_error) * min_mass / self.precision)
        to_int = math.floor((1 + self.max_error) * max_mass / self.precision)
        if from_int > 9223372036854775807 or to_int > 9223372036854775807:
            raise ArithmeticError(
                f"Mass range ({min_mass} - {max_mass}) too large to "
                f"decompose with current precision: {self.precision}"
            )
        start = max(0, int(from_int))
        end = max(start, int(to_int))
        return start, end

    def _populate_counts(
        self,
        user_max_counts: Optional[dict[str, int]],
        user_min_counts: Optional[dict[str, int]],
    ) -> tuple[dict[str, int], dict[str, int]]:
        final_min_counts: dict[str, int] = {
            e.symbol: 0 for e in self.elements
        }
        if user_min_counts:
            for element, count in user_min_counts.items():
                if count == float('inf'):
                    user_min_counts[element] = 0
            final_min_counts.update(**user_min_counts)

        final_max_counts: dict[str, float] = {
            e.symbol: float('inf') for e in self.elements
        }
        if user_max_counts:
            final_max_counts.update(**user_max_counts)

        return final_max_counts, final_min_counts


# ============================================================================
# FormulaValidator (from core/validator.py — isotope matching stripped)
# ============================================================================

class FormulaValidator:
    """Validates molecular formulae against chemical rules."""

    @staticmethod
    def validate_rdbe(
        formula: Union[Formula, LightFormula],
        min_rdbe: float,
        max_rdbe: float,
    ) -> bool:
        rdbe = get_rdbe(formula)
        if rdbe is None:
            return False
        return min_rdbe <= rdbe <= max_rdbe

    def validate(
        self,
        formula: Union[Formula, LightFormula],
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
    ) -> tuple[bool, None]:
        if filter_rdbe is not None:
            min_rdbe, max_rdbe = filter_rdbe
            if not self.validate_rdbe(formula, min_rdbe, max_rdbe):
                return False, None
        if check_octet:
            if not passes_octet_rule(formula):
                return False, None
        return True, None


# ============================================================================
# FormulaCandidate and FormulaSearchResults
# ============================================================================

@dataclass(slots=True)
class FormulaCandidate:
    """
    Structured result from formula finding.

    Attributes:
        formula: The molecular formula
        error_ppm: Mass error in parts per million
        error_da: Mass error in Daltons
        rdbe: Ring and Double Bond Equivalents (may be None)
    """
    formula: Union[Formula, LightFormula]
    error_ppm: float
    error_da: float
    rdbe: Optional[float]


@dataclass
class FormulaSearchResults:
    """
    Container for formula search results with filtering and display.

    Supports iteration, indexing, slicing, and formatted display.
    """
    candidates: list[FormulaCandidate]
    query_mass: float
    query_params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    @overload
    def __getitem__(self, idx: int) -> FormulaCandidate: ...

    @overload
    def __getitem__(self, idx: slice) -> 'FormulaSearchResults': ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FormulaSearchResults(
                candidates=self.candidates[idx],
                query_mass=self.query_mass,
                query_params=self.query_params,
            )
        return self.candidates[idx]

    def __repr__(self) -> str:
        n = len(self.candidates)
        if n == 0:
            return (
                f"FormulaSearchResults(query_mass={self.query_mass:.4f}, "
                f"n_results=0)"
            )
        lines = [
            f"FormulaSearchResults(query_mass={self.query_mass:.4f}, "
            f"n_results={n})",
            "",
            self.to_table(max_rows=5),
        ]
        return "\n".join(lines)

    def to_table(self, max_rows: Optional[int] = None) -> str:
        if len(self.candidates) == 0:
            return "No candidates found."

        candidates_to_show = self.candidates
        if max_rows is not None:
            candidates_to_show = self.candidates[:max_rows]

        header = f"{'Formula':<25} {'Error (ppm)':<15} {'Error (Da)':<15} {'RDBE':<10}"
        lines: list[str] = [header, "-" * 70]

        for c in candidates_to_show:
            formula_str = c.formula.formula
            rdbe_str = f"{c.rdbe:.1f}" if c.rdbe is not None else "N/A"
            row = (
                f"{formula_str:<25} {c.error_ppm:>14.2f} "
                f"{c.error_da:>14.6f} {rdbe_str:>9}"
            )
            lines.append(row)

        if max_rows is not None and len(self.candidates) > max_rows:
            lines.append(f"... and {len(self.candidates) - max_rows} more")

        return "\n".join(lines)

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        data = []
        for c in self.candidates:
            data.append({
                'formula': c.formula.formula,
                'error_ppm': c.error_ppm,
                'error_da': c.error_da,
                'rdbe': c.rdbe,
                'mass': c.formula.monoisotopic_mass,
            })
        return pd.DataFrame(data)

    def filter_by_rdbe(
        self, min_rdbe: float, max_rdbe: float,
    ) -> 'FormulaSearchResults':
        filtered = [
            c for c in self.candidates
            if c.rdbe is not None and min_rdbe <= c.rdbe <= max_rdbe
        ]
        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={**self.query_params, 'filter_rdbe': (min_rdbe, max_rdbe)},
        )

    def filter_by_octet(self) -> 'FormulaSearchResults':
        filtered = [
            c for c in self.candidates
            if passes_octet_rule(c.formula)
        ]
        return FormulaSearchResults(
            candidates=filtered,
            query_mass=self.query_mass,
            query_params={**self.query_params, 'check_octet': True},
        )

    def filter_by_error(
        self,
        max_ppm: Optional[float] = None,
        max_da: Optional[float] = None,
    ) -> 'FormulaSearchResults':
        if max_ppm is None and max_da is None:
            raise ValueError("At least one of max_ppm or max_da must be specified")
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
                'max_error_da': max_da,
            },
        )

    def top(self, n: int = 10) -> 'FormulaSearchResults':
        return FormulaSearchResults(
            candidates=self.candidates[:n],
            query_mass=self.query_mass,
            query_params=self.query_params,
        )


# ============================================================================
# FormulaFinder — main API (from core/finder.py)
# ============================================================================

class FormulaFinder:
    """
    API for finding molecular formulae from accurate masses.

    Initialize once with a set of elements, then call find_formulae() for
    each query mass.

    Example:
        >>> finder = FormulaFinder('CHNOPS')
        >>> results = finder.find_formulae(mass=180.063, error_ppm=5.0)
        >>> print(results)
    """

    def __init__(
        self,
        elements: Iterable[str] = 'CHNOPS',
        use_precalculated: bool = True,
    ):
        self.decomposer = MassDecomposer(
            elements=elements,
            use_precalculated=use_precalculated,
        )
        self.validator = FormulaValidator()

        self._symbols = list(self.decomposer.element_symbols)
        self._has_known_bond_e = all(s in BOND_ELECTRONS for s in self._symbols)
        if self._has_known_bond_e:
            self._rdbe_coeffs = np.array(
                [0.5 * (BOND_ELECTRONS[s] - 2) for s in self._symbols],
                dtype=np.float64,
            )
        else:
            self._rdbe_coeffs = None

    @staticmethod
    def _parse_adduct(adduct_str: str) -> tuple[Formula, float]:
        if '+' in adduct_str:
            raise ValueError(
                "Adduct string must not contain '+'. "
                "Specify charge separately using the 'charge' parameter."
            )
        if adduct_str.startswith('-'):
            adduct_formula_str = adduct_str[1:]
            adduct_formula = Formula(adduct_formula_str)
            adduct_mass = -adduct_formula.monoisotopic_mass
        else:
            adduct_formula = Formula(adduct_str)
            adduct_mass = adduct_formula.monoisotopic_mass
        return adduct_formula, adduct_mass

    def find_formulae(
        self,
        mass: float,
        charge: int = 0,
        error_ppm: Optional[float] = 0.0,
        error_da: Optional[float] = 0.0,
        adduct: Optional[str] = None,
        min_counts: Optional[dict[str, int] | str] = None,
        max_counts: Optional[dict[str, int] | str] = None,
        max_results: int = 10000,
        filter_rdbe: Optional[tuple[float, float]] = None,
        check_octet: bool = False,
    ) -> FormulaSearchResults:
        """
        Find molecular formula candidates for a given mass.

        Args:
            mass: Target mass to decompose (m/z value)
            charge: Charge state of the ion (default: 0, neutral)
            error_ppm: Mass tolerance in parts per million
            error_da: Mass tolerance in Daltons
            adduct: Neutral adduct formula string.
                Examples: "Na" for [M+Na]+, "H" for [M+H]+, "-H" for [M-H]-
            min_counts: Minimum element counts (dict or formula string)
            max_counts: Maximum element counts (dict or formula string)
            max_results: Maximum candidates before filtering (default: 10000)
            filter_rdbe: (min_rdbe, max_rdbe) tuple for RDBE filtering
            check_octet: If True, enforce the octet rule

        Returns:
            FormulaSearchResults sorted by mass error (smallest first)
        """
        # Parse adduct
        adduct_formula = None
        adduct_mass = 0.0
        if adduct:
            adduct_formula, adduct_mass = self._parse_adduct(adduct)

        # Parse min/max counts
        min_counts_dict: dict[str, int] | None = None
        if isinstance(min_counts, dict):
            min_counts_dict = min_counts
        elif min_counts is not None:
            min_counts_dict = to_bounds_dict(
                min_counts,
                elements=[x.symbol for x in self.decomposer.elements],
            )

        max_counts_dict: dict[str, int] | None = None
        if isinstance(max_counts, dict):
            max_counts_dict = max_counts
        elif max_counts is not None:
            max_counts_dict = to_bounds_dict(
                max_counts,
                elements=[x.symbol for x in self.decomposer.elements],
            )

        adjusted_mass = mass - adduct_mass

        symbols = self._symbols
        has_known_bond_e = self._has_known_bond_e
        can_prefilter = (
            adduct is None
            and (filter_rdbe is not None or check_octet)
            and has_known_bond_e
        )

        if has_known_bond_e:
            rdbe_coeffs = self._rdbe_coeffs

        # Push RDBE/octet into Numba when possible
        decompose_kwargs = {}
        if can_prefilter:
            rdbe_min = filter_rdbe[0] if filter_rdbe is not None else -np.inf
            rdbe_max = filter_rdbe[1] if filter_rdbe is not None else np.inf
            decompose_kwargs = {
                'rdbe_coeffs': rdbe_coeffs,
                'rdbe_min': float(rdbe_min),
                'rdbe_max': float(rdbe_max),
                'check_octet': check_octet,
                'charge_parity_even': abs(charge) % 2 == 0,
            }

        counts, symbols = self.decomposer.decompose_to_counts(
            query_mass=adjusted_mass,
            charge=charge,
            ppm_error=error_ppm,
            mz_error=error_da,
            min_counts=min_counts_dict,
            max_counts=max_counts_dict,
            max_results=max_results,
            **decompose_kwargs,
        )

        if can_prefilter:
            remaining_filter_rdbe = None
            remaining_check_octet = False
        else:
            remaining_filter_rdbe = filter_rdbe
            remaining_check_octet = check_octet

        # Vectorized error and RDBE computation
        charge_mass_offset = ELECTRON.mass * charge
        real_masses_arr = self.decomposer.real_masses

        if len(counts) > 0:
            counts_f = counts.astype(np.float64)
            exact_masses = counts_f @ real_masses_arr - charge_mass_offset
            if adduct_formula is not None:
                exact_masses += adduct_formula.monoisotopic_mass
            all_err_ppm = (exact_masses - mass) / mass * 1e6
            if has_known_bond_e:
                all_rdbes = counts_f @ rdbe_coeffs + 1.0
            else:
                all_rdbes = None
            sort_order = np.argsort(np.abs(all_err_ppm))
        else:
            sort_order = np.array([], dtype=np.intp)

        needs_validation = (
            remaining_filter_rdbe is not None
            or remaining_check_octet
        )

        candidates: list[FormulaCandidate] = []

        # Pre-extract adduct element counts
        if adduct_formula is not None:
            adduct_elements = {}
            for sym, item in adduct_formula.composition().items():
                if sym == '' or sym == 'e-':
                    continue
                if item.count > 0:
                    adduct_elements[sym] = item.count
            adduct_charge = adduct_formula.charge
        else:
            adduct_elements = None
            adduct_charge = 0

        # Batch-convert numpy arrays to Python lists
        if len(sort_order) > 0:
            sorted_counts_list = counts[sort_order].tolist()
            sorted_masses = exact_masses[sort_order].tolist()
            sorted_err_ppm = all_err_ppm[sort_order].tolist()
            if all_rdbes is not None:
                sorted_rdbes = all_rdbes[sort_order].tolist()
            else:
                sorted_rdbes = None

        combined_charge = charge + adduct_charge
        n_rows = len(sort_order)
        append_candidate = candidates.append

        if adduct_elements is None:
            for idx in range(n_rows):
                row_list = sorted_counts_list[idx]

                formula = LightFormula.from_counts(
                    symbols=symbols,
                    counts=row_list,
                    charge=combined_charge,
                    monoisotopic_mass=sorted_masses[idx],
                )

                if needs_validation:
                    passes, _ = self.validator.validate(
                        formula=formula,
                        filter_rdbe=remaining_filter_rdbe,
                        check_octet=remaining_check_octet,
                    )
                    if not passes:
                        continue

                append_candidate(
                    FormulaCandidate(
                        formula=formula,
                        error_ppm=sorted_err_ppm[idx],
                        error_da=sorted_masses[idx] - mass,
                        rdbe=sorted_rdbes[idx] if sorted_rdbes is not None else None,
                    )
                )
        else:
            n_symbols = len(symbols)
            for idx in range(n_rows):
                row_list = sorted_counts_list[idx]

                elements_dict: dict[str, int] = {}
                for j in range(n_symbols):
                    c = row_list[j]
                    if c > 0:
                        elements_dict[symbols[j]] = c

                for sym, cnt in adduct_elements.items():
                    elements_dict[sym] = elements_dict.get(sym, 0) + cnt

                formula = LightFormula(
                    elements=elements_dict,
                    charge=combined_charge,
                    monoisotopic_mass=sorted_masses[idx],
                )

                if needs_validation:
                    passes, _ = self.validator.validate(
                        formula=formula,
                        filter_rdbe=remaining_filter_rdbe,
                        check_octet=remaining_check_octet,
                    )
                    if not passes:
                        continue

                append_candidate(
                    FormulaCandidate(
                        formula=formula,
                        error_ppm=sorted_err_ppm[idx],
                        error_da=sorted_masses[idx] - mass,
                        rdbe=sorted_rdbes[idx] if sorted_rdbes is not None else None,
                    )
                )

        query_params = {
            'mass': mass, 'charge': charge,
            'error_ppm': error_ppm, 'error_da': error_da,
            'adduct': adduct, 'min_counts': min_counts,
            'max_counts': max_counts, 'max_results': max_results,
            'filter_rdbe': filter_rdbe, 'check_octet': check_octet,
        }

        return FormulaSearchResults(
            candidates=candidates,
            query_mass=mass,
            query_params=query_params,
        )

    @property
    def element_set(self) -> set[str]:
        return set(self.decomposer.element_symbols)


# ============================================================================
# Convenience function
# ============================================================================

_CHNOPS_FINDER: Optional[FormulaFinder] = None


def find_chnops(
    mass: float,
    error_ppm: float = 5.0,
    **kwargs,
) -> FormulaSearchResults:
    """
    Quick convenience function using a shared CHNOPS finder.

    >>> results = find_chnops(180.063, error_ppm=5.0)
    >>> print(results)
    """
    global _CHNOPS_FINDER
    if _CHNOPS_FINDER is None:
        _CHNOPS_FINDER = FormulaFinder('CHNOPS')
    return _CHNOPS_FINDER.find_formulae(mass=mass, error_ppm=error_ppm, **kwargs)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python find_formulae.py <mass> [ppm_error] [elements]")
        print("Example: python find_formulae.py 180.063 5.0 CHNOPS")
        sys.exit(1)

    query_mass = float(sys.argv[1])
    ppm = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    elems = sys.argv[3] if len(sys.argv) > 3 else 'CHNOPS'

    print(f"Finding formulae for mass={query_mass}, ppm={ppm}, elements={elems}")
    t0 = time.perf_counter()
    finder = FormulaFinder(elems)
    t1 = time.perf_counter()
    print(f"Initialization: {(t1 - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    results = finder.find_formulae(mass=query_mass, error_ppm=ppm)
    t1 = time.perf_counter()
    print(f"Search: {(t1 - t0) * 1000:.2f} ms")
    print()
    print(results)
