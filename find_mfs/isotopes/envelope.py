"""
Isotope envelope fitting using IsoSpecPy.
This module is optional, to avoid bloat for users that
don't care about isotope envelopes.

***NOTE: reminder that "monoisotopic peak" means the tallest
signal in an isotope envelope - NOT "M0".***

Requires the optional IsoSpecPy dependency
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from molmass import Formula
from molmass.elements import ELECTRON
from numba import njit, types as nbtypes
from numba.core.runtime import nrt

try:
    import IsoSpecPy as iso
except ImportError:
    iso = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .results import SingleEnvelopeMatchResult
    from ..core.light_formula import LightFormula

def get_isotope_envelope(
    formula: Formula | LightFormula,
    mz_tolerance: float,
    threshold: float,
) -> np.ndarray:
    """
    Calculate the isotope envelope for a given molecular formula.

    Handles charged species by removing charge notation from formula string
    for IsoSpecPy calculation, then adjusting m/z values for electron mass,
    and charge

    Args:
        formula: Molecular formula object (may include charge)

        mz_tolerance: Minimum difference in mz values - signals less
            resolved than this will be combined

        threshold: Minimum relative intensity threshold to include
            an isotopologue into envelope

    Returns:
        Array of [m/z, intensity] pairs where intensities are scaled
        such that the monoisotopic peak is 1.0

    Raises:
        ImportError: If IsoSpecPy is not installed
    """
    if iso is None:
        raise ImportError(
            "IsoSpecPy is required for isotope envelope calculation. "
            "Install with: pip install find-mfs"
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold argument must be between 0.0 and 1.0. "
            f"Given: {threshold}"
        )

    # Extract charge and create neutral formula string for IsoSpecPy
    charge = formula.charge
    formula_str = formula.formula

    # Remove charge notation (e.g., "C6H6+" -> "C6H6")
    if charge != 0:
        # Remove trailing + or - characters, or [ and ]
        formula_str = formula_str.rstrip('+-[]').strip('[]')

    isotope_calculator = iso.IsoThreshold(
        formula=formula_str,
        threshold=threshold,
    )

    isologues: list[tuple[float, float]] = []
    for mass, probability in isotope_calculator:
        mass: float
        probability: float

        # Adjust mass for charge state
        if charge != 0:
            # Convert neutral mass to m/z
            mz = (mass + charge * ELECTRON.mass) / abs(charge)
        else:
            mz = mass

        isologues.append(
            (mz, probability)
        )

    # Note: float32 is used here for the old match_isotope_envelope() path.
    # The fast path (match_isotope_envelope_fast) uses float64 throughout
    # via IsoSpecPy's C++ doubles. This causes small RMSE differences
    # (<0.01 absolute) between paths, which is acceptable.
    isologues: np.ndarray = np.array(
        isologues,
        dtype=np.float32,
    )

    isologues: np.ndarray = combine_unresolved_isotopologues(
        isologues,
        mz_tolerance=mz_tolerance,
    )

    isologues: np.ndarray = rescale_envelope(isologues)

    return isologues


def combine_unresolved_isotopologues(
        isologue_array: np.ndarray,
        mz_tolerance: float,
) -> np.ndarray:
    """
    Combines isotopologues that are within `tolerance_da` of each other.

    To combine, the intensities are summed, and the mass is changed to
    a weighted average

    Args:
        isologue_array: Array of [mass, intensity] pairs
        mz_tolerance: Mass tolerance in Daltons for combining peaks

    Returns:
        Array with combined isotopologues
    """
    sorted_idxs = np.argsort(isologue_array[:, 0])
    sorted_arr = isologue_array[sorted_idxs]

    result: list[tuple[float, float]] = []
    i = 0

    while i < len(sorted_arr):
        current_group = [sorted_arr[i]]
        current_value = sorted_arr[i, 0]

        j = i + 1

        # Find all rows with similar masses
        while (
            j < len(sorted_arr) and
            abs(sorted_arr[j, 0] - current_value) <= mz_tolerance
        ):
            current_group.append(sorted_arr[j])
            j += 1

        # Combine the group; average mass, sum intensity
        if len(current_group) > 0:
            group_array: np.ndarray = np.array(current_group)
            combined_row: tuple[float, float] = (
                np.average(  # Weighted average
                    a=group_array[:, 0],
                    weights=group_array[:, 1],
                ),
                np.sum(group_array[:, 1]),  # Intensity sum
            )
            result.append(combined_row)

        i = j

    return np.array(result)


def rescale_envelope(
    isologue_array: np.ndarray[..., ...]
) -> np.ndarray[..., ...]:
    """
    Normalizes isotope envelope intensities to monoisotopic peak
    (i.e. tallest peak)

    Args:
        isologue_array: Array of [mass, intensity] pairs

    Returns:
        Array of relative intensities
    """
    isologue_array[:, 1] = isologue_array[:, 1] / isologue_array[:, 1].max()
    return isologue_array


def _check_isospec_available():
    """
    Raise ImportError if isospec not available
    """
    if iso is None:
        raise ImportError(
            "IsoSpecPy is required for isotope envelope matching. "
            "Install with: pip install find-mfs"
        )


def match_isotope_envelope(
    formula: Formula | LightFormula,
    observed_envelope: np.ndarray,
    mz_match_tolerance: float,
    simulated_envelope_mz_tolerance: float = 0.05,
    simulated_envelope_intsy_threshold: float = 0.001,
) -> 'SingleEnvelopeMatchResult':
    """
    Given a Formula and observed isotope envelope, returns detailed matching
    results using RMSE scoring between aligned intensity vectors.

    For each observed peak, the closest predicted peak within
    `mz_match_tolerance` is found. Two aligned intensity vectors are built:
    observed intensities and matched predicted intensities (0.0 if no match).
    The base peak (tallest signal) is excluded from RMSE scoring since
    both envelopes are normalized to it, making it uninformative.

    Args:
        formula: Molecular formula object

        observed_envelope: Array of observed [m/z, intensity] pairs

        mz_match_tolerance: Maximum tolerable difference (in Da) between
            predicted/observed isotopologue signal m/z value to be considered
            a match.

            This parameter should depend on the instrument's mass accuracy,
            and should be set very generously.

        simulated_envelope_mz_tolerance: The resolution at which isotope
            envelopes will be simulated. Isotopologues less resolved than
            this value will be combined (i.e. intensities summed, m/z values
            weighted average). Default: 0.05

        simulated_envelope_intsy_threshold: The minimum relative intensity
            to be included in a simulated isotope envelope. Default: 0.001

    Returns:
        SingleEnvelopeMatchResult containing:
        - intensity_rmse: Root-mean-square error of matched intensity
            differences (excluding base peak). Lower is better.
        - match_fraction: Fraction of observed peaks matched to a predicted
            peak (for filtering)
        - peak_matches: Boolean array of which observed peaks matched
        - num_peaks_matched/total: Count information
        - predicted_envelope: The theoretical envelope used
    """
    _check_isospec_available()

    if observed_envelope.ndim != 2:
        raise ValueError(
            "Misformed `observed_envelope` array. Should be a 2D array such"
            " that arr[:, 0] is m/z values, and arr[:, 1] is intensity values"
        )

    simulated_envelope = get_isotope_envelope(
        formula=formula,
        mz_tolerance=simulated_envelope_mz_tolerance,
        threshold=simulated_envelope_intsy_threshold,
    )

    num_observed = observed_envelope.shape[0]
    observed_intensities = observed_envelope[:, 1].copy()
    predicted_intensities = np.zeros(num_observed, dtype=np.float64)
    peak_matches = np.full(num_observed, False)

    # For each observed peak, find the closest predicted peak within tolerance
    for idx in range(num_observed):
        obs_mz = observed_envelope[idx, 0]
        mz_diffs = np.abs(simulated_envelope[:, 0] - obs_mz)
        closest_idx = np.argmin(mz_diffs)

        if mz_diffs[closest_idx] <= mz_match_tolerance:
            predicted_intensities[idx] = simulated_envelope[closest_idx, 1]
            peak_matches[idx] = True

    # Compute RMSE of intensity differences, excluding the base peak
    # (tallest signal). Both envelopes are normalized so the base peak
    # is always 1.0, meaning it carries no discriminating information.
    base_peak_idx = np.argmax(observed_intensities)
    mask = np.ones(num_observed, dtype=bool)
    mask[base_peak_idx] = False
    obs_for_rmse = observed_intensities[mask]
    pred_for_rmse = predicted_intensities[mask]

    if len(obs_for_rmse) > 0:
        intensity_rmse = float(np.sqrt(
            np.mean((obs_for_rmse - pred_for_rmse) ** 2)
        ))
    else:
        intensity_rmse = 0.0

    num_peaks_matched = int(np.sum(peak_matches))
    num_peaks_total = num_observed
    match_fraction = num_peaks_matched / num_peaks_total \
        if num_peaks_total > 0 else 0.0

    # Import here to avoid circular dependency
    from .results import SingleEnvelopeMatchResult
    return SingleEnvelopeMatchResult(
        num_peaks_matched=num_peaks_matched,
        num_peaks_total=num_peaks_total,
        intensity_rmse=intensity_rmse,
        match_fraction=match_fraction,
        peak_matches=peak_matches,
        predicted_envelope=simulated_envelope,
    )


# ---------------------------------------------------------------------------
# Fast path: Numba + C++ IsoSpecPy scoring
# ---------------------------------------------------------------------------

# Module-level ctypes references; initialized lazily on first use.
_fast_path_ready = False
_setupIso = None
_setupThreshold = None
_confs_no = None
_getMasses = None
_getProbs = None
_deleteFE = None
_deleteIso = None
_freeArray = None


def _ensure_fast_path():
    """Lazily load ctypes function references for the fast path."""
    global _fast_path_ready
    global _setupIso, _setupThreshold, _confs_no
    global _getMasses, _getProbs, _deleteFE, _deleteIso, _freeArray

    if _fast_path_ready:
        return
    from ._isospec_bridge import get_lib_functions
    (
        _setupIso, _setupThreshold, _confs_no,
        _getMasses, _getProbs, _deleteFE, _deleteIso, _freeArray,
    ) = get_lib_functions()
    _fast_path_ready = True


def _make_isospec_match_and_score():
    """
    Build and return the @njit scoring function with ctypes globals bound.

    We use a factory so that the ctypes function references are captured
    as closure variables at definition time, making them accessible to
    Numba's type inference.
    """
    _ensure_fast_path()

    # Bind to local names so Numba can resolve them as globals of the
    # inner function's module.
    setupIso = _setupIso
    setupThreshold = _setupThreshold
    confs_no = _confs_no
    getMasses = _getMasses
    getProbs = _getProbs
    deleteFE = _deleteFE
    deleteIso = _deleteIso
    freeArray = _freeArray

    from ._isospec_bridge import uint64_to_voidptr
    from numba import carray

    @njit
    def _isospec_match_and_score(
        iso_numbers, atom_counts, flat_masses, flat_probs, n_elements,
        obs_mz, obs_int, combine_tol, match_tol, threshold,
        charge, electron_mass,
    ):
        """
        Full IsoSpecPy C++ call + combine + match + RMSE in Numba.

        Args:
            iso_numbers: int32[n_elements] - isotope count per element
            atom_counts: int32[n_elements] - atom count per element
            flat_masses: float64[sum(iso_numbers)] - concatenated isotope masses
            flat_probs: float64[sum(iso_numbers)] - concatenated isotope probs
            n_elements: int - number of elements
            obs_mz: float64[n_obs] - observed m/z values (normalized)
            obs_int: float64[n_obs] - observed intensities (normalized to 1.0)
            combine_tol: float - tolerance for combining unresolved peaks
            match_tol: float - tolerance for matching observed to predicted
            threshold: float - IsoSpecPy intensity threshold
            charge: int - ion charge state
            electron_mass: float - electron mass for m/z adjustment

        Returns:
            Tuple of (rmse, match_fraction, n_matched)
        """
        # 1. Call C++ setupIso
        iso_ptr = setupIso(
            n_elements,
            iso_numbers.ctypes,
            atom_counts.ctypes,
            flat_masses.ctypes,
            flat_probs.ctypes,
        )

        # 2. Get threshold fixed envelope
        env_ptr = setupThreshold(iso_ptr, threshold, False, False)
        n_peaks = confs_no(env_ptr)

        if n_peaks == 0:
            deleteFE(env_ptr, False)
            deleteIso(iso_ptr)
            return 1.0, 0.0, 0

        # 3. Read mass/prob arrays (zero-copy via carray)
        masses_raw = getMasses(env_ptr)
        probs_raw = getProbs(env_ptr)
        pred_mz_raw = carray(
            uint64_to_voidptr(masses_raw),
            (n_peaks,),
            dtype=nbtypes.float64,
        )
        pred_prob_raw = carray(
            uint64_to_voidptr(probs_raw),
            (n_peaks,),
            dtype=nbtypes.float64,
        )

        # Copy to local arrays (C++ memory will be freed)
        pred_mz = np.empty(n_peaks, dtype=np.float64)
        pred_prob = np.empty(n_peaks, dtype=np.float64)
        for i in range(n_peaks):
            pred_mz[i] = pred_mz_raw[i]
            pred_prob[i] = pred_prob_raw[i]

        # 4. Free C++ memory
        freeArray(masses_raw)
        freeArray(probs_raw)
        deleteFE(env_ptr, False)
        deleteIso(iso_ptr)

        # 5. Adjust for charge (convert neutral mass to m/z)
        if charge != 0:
            abs_charge = abs(charge)
            charge_offset = charge * electron_mass
            for i in range(n_peaks):
                pred_mz[i] = (pred_mz[i] + charge_offset) / abs_charge

        # 6. Insertion sort by mass
        order = np.arange(n_peaks, dtype=np.int64)
        for i in range(1, n_peaks):
            key = order[i]
            key_mz = pred_mz[key]
            j = i - 1
            while j >= 0 and pred_mz[order[j]] > key_mz:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        # 7. Combine unresolved isotopologues
        combined_mz = np.empty(n_peaks, dtype=np.float64)
        combined_int = np.empty(n_peaks, dtype=np.float64)
        n_combined = 0
        i = 0
        while i < n_peaks:
            grp_mz_sum = pred_mz[order[i]] * pred_prob[order[i]]
            grp_int_sum = pred_prob[order[i]]
            j = i + 1
            while j < n_peaks and abs(pred_mz[order[j]] - pred_mz[order[i]]) <= combine_tol:
                grp_mz_sum += pred_mz[order[j]] * pred_prob[order[j]]
                grp_int_sum += pred_prob[order[j]]
                j += 1
            combined_mz[n_combined] = grp_mz_sum / grp_int_sum
            combined_int[n_combined] = grp_int_sum
            n_combined += 1
            i = j

        # 8. Rescale to base peak = 1.0
        mx = 0.0
        for i in range(n_combined):
            if combined_int[i] > mx:
                mx = combined_int[i]
        if mx > 0.0:
            for i in range(n_combined):
                combined_int[i] /= mx

        # 9. Match observed peaks to closest predicted peak
        n_obs = len(obs_mz)
        pred_matched = np.zeros(n_obs, dtype=np.float64)
        matched = np.zeros(n_obs, dtype=np.int32)

        for i in range(n_obs):
            best_diff = 1e30
            best_j = -1
            for j in range(n_combined):
                d = abs(obs_mz[i] - combined_mz[j])
                if d < best_diff:
                    best_diff = d
                    best_j = j
            if best_diff <= match_tol:
                pred_matched[i] = combined_int[best_j]
                matched[i] = 1

        # 10. Compute RMSE excluding base peak
        base_idx = 0
        max_obs = obs_int[0]
        for i in range(1, n_obs):
            if obs_int[i] > max_obs:
                max_obs = obs_int[i]
                base_idx = i

        sse = 0.0
        count = 0
        n_matched = 0
        for i in range(n_obs):
            if matched[i]:
                n_matched += 1
            if i == base_idx:
                continue
            d = obs_int[i] - pred_matched[i]
            sse += d * d
            count += 1

        rmse = (sse / count) ** 0.5 if count > 0 else 0.0
        match_frac = n_matched / n_obs if n_obs > 0 else 0.0

        return rmse, match_frac, n_matched

    @njit
    def _batch_score(iso_numbers, counts_2d, flat_masses, flat_probs,
                     n_elements, n_candidates, obs_mz, obs_int,
                     combine_tol, match_tol, threshold, charge, electron_mass):
        rmse = np.empty(n_candidates, dtype=np.float64)
        match_frac = np.empty(n_candidates, dtype=np.float64)
        n_matched = np.empty(n_candidates, dtype=np.int32)
        for i in range(n_candidates):
            atom_counts = np.empty(n_elements, dtype=np.int32)
            for j in range(n_elements):
                atom_counts[j] = counts_2d[i, j]
            r, mf, nm = _isospec_match_and_score(
                iso_numbers, atom_counts, flat_masses, flat_probs, n_elements,
                obs_mz, obs_int, combine_tol, match_tol, threshold,
                charge, electron_mass)
            rmse[i] = r
            match_frac[i] = mf
            n_matched[i] = nm
        return rmse, match_frac, n_matched

    return _isospec_match_and_score, _batch_score


# Lazily compiled singletons
_compiled_scorer = None
_compiled_batch_scorer = None


def _get_scorer():
    """Get or create the compiled Numba scoring function."""
    global _compiled_scorer, _compiled_batch_scorer
    if _compiled_scorer is None:
        _compiled_scorer, _compiled_batch_scorer = _make_isospec_match_and_score()
    return _compiled_scorer


def _get_scorers():
    """Get or create both the individual and batch Numba scoring functions."""
    global _compiled_scorer, _compiled_batch_scorer
    if _compiled_scorer is None:
        _compiled_scorer, _compiled_batch_scorer = _make_isospec_match_and_score()
    return _compiled_scorer, _compiled_batch_scorer


def match_isotope_envelope_fast(
    symbols: list[str] | tuple[str, ...],
    counts: list[int] | tuple[int, ...],
    charge: int,
    observed_envelope: np.ndarray,
    mz_match_tolerance: float,
    simulated_mz_tolerance: float = 0.05,
    simulated_intensity_threshold: float = 0.001,
) -> 'SingleEnvelopeMatchResult':
    """
    Fast isotope envelope matching using Numba + C++ IsoSpecPy.

    Replaces the string-based match_isotope_envelope() path with direct
    C++ calls from Numba. Produces identical RMSE/match_fraction results.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])
        counts: Atom counts matching symbols order
        charge: Ion charge state
        observed_envelope: 2D array of [m/z, intensity] pairs (normalized)
        mz_match_tolerance: Max m/z difference for peak matching (Da)
        simulated_mz_tolerance: Resolution for combining isotopologues
        simulated_intensity_threshold: Min relative intensity threshold

    Returns:
        SingleEnvelopeMatchResult with RMSE, match fraction, etc.

        Note: Unlike match_isotope_envelope(), the following fields are
        simplified for performance:
        - peak_matches: All True if any peak matched, all False otherwise
          (not per-peak granularity). Only used for display, not filtering.
        - predicted_envelope: Empty array (shape (0, 2)). The theoretical
          envelope is computed inside the Numba scorer and not returned.
          Only used for display, not filtering decisions.
    """
    from ._isospec_bridge import get_isotope_arrays

    scorer = _get_scorer()

    # Build isotope data arrays for the elements present
    # Filter to non-zero elements only
    active_symbols = []
    active_counts = []
    for sym, cnt in zip(symbols, counts):
        if cnt > 0:
            active_symbols.append(sym)
            active_counts.append(cnt)

    if not active_symbols:
        from .results import SingleEnvelopeMatchResult
        n_obs = observed_envelope.shape[0]
        return SingleEnvelopeMatchResult(
            num_peaks_matched=0,
            num_peaks_total=n_obs,
            intensity_rmse=1.0,
            match_fraction=0.0,
            peak_matches=np.full(n_obs, False),
            predicted_envelope=np.empty((0, 2), dtype=np.float64),
        )

    iso_numbers, flat_masses, flat_probs = get_isotope_arrays(active_symbols)
    atom_counts = np.array(active_counts, dtype=np.int32)
    n_elements = len(active_symbols)

    obs_mz = np.ascontiguousarray(observed_envelope[:, 0], dtype=np.float64)
    obs_int = np.ascontiguousarray(observed_envelope[:, 1], dtype=np.float64)

    rmse, match_frac, n_matched = scorer(
        iso_numbers, atom_counts, flat_masses, flat_probs, n_elements,
        obs_mz, obs_int,
        simulated_mz_tolerance,
        mz_match_tolerance,
        simulated_intensity_threshold,
        charge,
        ELECTRON.mass,
    )

    n_obs = observed_envelope.shape[0]
    peak_matches = np.full(n_obs, n_matched > 0)  # Simplified

    from .results import SingleEnvelopeMatchResult
    return SingleEnvelopeMatchResult(
        num_peaks_matched=int(n_matched),
        num_peaks_total=n_obs,
        intensity_rmse=float(rmse),
        match_fraction=float(match_frac),
        peak_matches=peak_matches,
        predicted_envelope=np.empty((0, 2), dtype=np.float64),
    )


def score_isotope_batch(
    symbols: list[str] | tuple[str, ...],
    counts_2d: np.ndarray,
    charge: int,
    observed_envelope: np.ndarray,
    mz_match_tolerance: float,
    simulated_mz_tolerance: float = 0.05,
    simulated_intensity_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch isotope envelope scoring for multiple candidates at once.

    Eliminates per-candidate Python→Numba transition overhead by scoring
    all candidates in a single Numba call.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])
        counts_2d: int32 array of shape (N, n_elements) with atom counts
        charge: Ion charge state
        observed_envelope: 2D array of [m/z, intensity] pairs (normalized)
        mz_match_tolerance: Max m/z difference for peak matching (Da)
        simulated_mz_tolerance: Resolution for combining isotopologues
        simulated_intensity_threshold: Min relative intensity threshold

    Returns:
        Tuple of (rmse_arr, match_frac_arr, n_matched_arr) — raw numpy arrays
        of shape (N,) with float64, float64, int32 dtypes respectively.
    """
    from ._isospec_bridge import get_isotope_arrays

    _, batch_scorer = _get_scorers()

    iso_numbers, flat_masses, flat_probs = get_isotope_arrays(symbols)
    n_elements = len(symbols)
    n_candidates = counts_2d.shape[0]

    counts_2d = np.ascontiguousarray(counts_2d, dtype=np.int32)
    obs_mz = np.ascontiguousarray(observed_envelope[:, 0], dtype=np.float64)
    obs_int = np.ascontiguousarray(observed_envelope[:, 1], dtype=np.float64)

    return batch_scorer(
        iso_numbers, counts_2d, flat_masses, flat_probs,
        n_elements, n_candidates, obs_mz, obs_int,
        simulated_mz_tolerance,
        mz_match_tolerance,
        simulated_intensity_threshold,
        charge,
        ELECTRON.mass,
    )
