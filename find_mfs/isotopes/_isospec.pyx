# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython IsoSpec bridge — replaces the Numba ctypes path with C-level
function pointer calls via dlopen/dlsym at module init.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
from posix.dlfcn cimport dlopen, dlsym, dlclose, dlerror, RTLD_LAZY

ctypedef double float64_t

# C function pointer types matching IsoSpec's C API
ctypedef void* (*setupIso_t)(int, const int32_t*, const int32_t*, const double*, const double*) noexcept nogil
ctypedef void* (*setupThresholdFixedEnvelope_t)(void*, double, bint, bint) noexcept nogil
ctypedef size_t (*confs_noFixedEnvelope_t)(void*) noexcept nogil
ctypedef const double* (*massesFixedEnvelope_t)(void*) noexcept nogil
ctypedef const double* (*probsFixedEnvelope_t)(void*) noexcept nogil
ctypedef void (*deleteFixedEnvelope_t)(void*, bint) noexcept nogil
ctypedef void (*deleteIso_t)(void*) noexcept nogil
ctypedef void (*freeReleasedArray_t)(void*) noexcept nogil

# Module-level function pointers
cdef setupIso_t _setupIso = NULL
cdef setupThresholdFixedEnvelope_t _setupThreshold = NULL
cdef confs_noFixedEnvelope_t _confs_no = NULL
cdef massesFixedEnvelope_t _getMasses = NULL
cdef probsFixedEnvelope_t _getProbs = NULL
cdef deleteFixedEnvelope_t _deleteFE = NULL
cdef deleteIso_t _deleteIso = NULL
cdef freeReleasedArray_t _freeArray = NULL
cdef void* _lib_handle = NULL
cdef bint _loaded = False


def _load_isospec_lib():
    """Load IsoSpec shared library and resolve function pointers."""
    global _setupIso, _setupThreshold, _confs_no
    global _getMasses, _getProbs, _deleteFE, _deleteIso, _freeArray
    global _lib_handle, _loaded

    if _loaded:
        return

    # Get the path to IsoSpec's shared library
    try:
        from IsoSpecPy.isoFFI import isoFFI
        lib_path = str(isoFFI.libpath)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot find IsoSpecPy C++ library: {e}. "
            "Install IsoSpecPy with: pip install IsoSpecPy"
        ) from e

    # dlopen the library
    cdef bytes path_bytes = lib_path.encode('utf-8')
    _lib_handle = dlopen(path_bytes, RTLD_LAZY)
    if _lib_handle == NULL:
        err = dlerror()
        raise ImportError(
            f"Cannot load IsoSpecPy C++ library at {lib_path}: "
            f"{err.decode('utf-8') if err else 'unknown error'}"
        )

    # Resolve function pointers
    _setupIso = <setupIso_t>dlsym(_lib_handle, "setupIso")
    _setupThreshold = <setupThresholdFixedEnvelope_t>dlsym(_lib_handle, "setupThresholdFixedEnvelope")
    _confs_no = <confs_noFixedEnvelope_t>dlsym(_lib_handle, "confs_noFixedEnvelope")
    _getMasses = <massesFixedEnvelope_t>dlsym(_lib_handle, "massesFixedEnvelope")
    _getProbs = <probsFixedEnvelope_t>dlsym(_lib_handle, "probsFixedEnvelope")
    _deleteFE = <deleteFixedEnvelope_t>dlsym(_lib_handle, "deleteFixedEnvelope")
    _deleteIso = <deleteIso_t>dlsym(_lib_handle, "deleteIso")
    _freeArray = <freeReleasedArray_t>dlsym(_lib_handle, "freeReleasedArray")

    # Verify all resolved
    if (_setupIso == NULL or _setupThreshold == NULL or _confs_no == NULL or
        _getMasses == NULL or _getProbs == NULL or _deleteFE == NULL or
        _deleteIso == NULL or _freeArray == NULL):
        dlclose(_lib_handle)
        _lib_handle = NULL
        raise ImportError("Failed to resolve one or more IsoSpec C functions")

    _loaded = True


cdef (double, double, int) _score_single_envelope(
    int32_t* iso_numbers, int32_t* atom_counts,
    double* flat_masses, double* flat_probs,
    int n_elements,
    double* obs_mz, double* obs_int, int n_obs,
    double combine_tol, double match_tol, double threshold,
    int charge, double electron_mass,
) noexcept nogil:
    """
    Score a single candidate against the observed envelope.
    All C-level, no GIL, no Python objects.

    Returns (rmse, match_fraction, n_matched).
    """
    cdef void* iso_ptr
    cdef void* env_ptr
    cdef size_t n_peaks
    cdef const double* masses_raw
    cdef const double* probs_raw

    # 1. Call C++ setupIso
    iso_ptr = _setupIso(
        n_elements, iso_numbers, atom_counts,
        flat_masses, flat_probs,
    )

    # 2. Get threshold fixed envelope
    env_ptr = _setupThreshold(iso_ptr, threshold, False, False)
    n_peaks = _confs_no(env_ptr)

    if n_peaks == 0:
        _deleteFE(env_ptr, False)
        _deleteIso(iso_ptr)
        return (1.0, 0.0, 0)

    # 3. Read mass/prob arrays
    masses_raw = _getMasses(env_ptr)
    probs_raw = _getProbs(env_ptr)

    # Copy to local C arrays (C++ memory will be freed)
    cdef double* pred_mz = <double*>malloc(n_peaks * sizeof(double))
    cdef double* pred_prob = <double*>malloc(n_peaks * sizeof(double))
    if pred_mz == NULL or pred_prob == NULL:
        if pred_mz != NULL: free(pred_mz)
        if pred_prob != NULL: free(pred_prob)
        _freeArray(<void*>masses_raw)
        _freeArray(<void*>probs_raw)
        _deleteFE(env_ptr, False)
        _deleteIso(iso_ptr)
        return (1.0, 0.0, 0)

    cdef size_t i, j
    for i in range(n_peaks):
        pred_mz[i] = masses_raw[i]
        pred_prob[i] = probs_raw[i]

    # 4. Free C++ memory
    _freeArray(<void*>masses_raw)
    _freeArray(<void*>probs_raw)
    _deleteFE(env_ptr, False)
    _deleteIso(iso_ptr)

    # 5. Adjust for charge (convert neutral mass to m/z)
    cdef int abs_charge
    cdef double charge_offset
    if charge != 0:
        abs_charge = abs(charge)
        charge_offset = charge * electron_mass
        for i in range(n_peaks):
            pred_mz[i] = (pred_mz[i] + charge_offset) / abs_charge

    # 6. Insertion sort by mass
    cdef double key_mz, key_prob
    cdef int ki
    for ki in range(1, <int>n_peaks):
        key_mz = pred_mz[ki]
        key_prob = pred_prob[ki]
        j = ki - 1
        while <int>j >= 0 and pred_mz[j] > key_mz:
            pred_mz[j + 1] = pred_mz[j]
            pred_prob[j + 1] = pred_prob[j]
            j -= 1
        pred_mz[j + 1] = key_mz
        pred_prob[j + 1] = key_prob

    # 7. Combine unresolved isotopologues
    cdef double* combined_mz = <double*>malloc(n_peaks * sizeof(double))
    cdef double* combined_int = <double*>malloc(n_peaks * sizeof(double))
    if combined_mz == NULL or combined_int == NULL:
        free(pred_mz)
        free(pred_prob)
        if combined_mz != NULL: free(combined_mz)
        if combined_int != NULL: free(combined_int)
        return (1.0, 0.0, 0)

    cdef int n_combined = 0
    cdef double grp_mz_sum, grp_int_sum

    i = 0
    while i < n_peaks:
        grp_mz_sum = pred_mz[i] * pred_prob[i]
        grp_int_sum = pred_prob[i]
        j = i + 1
        while j < n_peaks and fabs(pred_mz[j] - pred_mz[i]) <= combine_tol:
            grp_mz_sum += pred_mz[j] * pred_prob[j]
            grp_int_sum += pred_prob[j]
            j += 1
        combined_mz[n_combined] = grp_mz_sum / grp_int_sum
        combined_int[n_combined] = grp_int_sum
        n_combined += 1
        i = j

    free(pred_mz)
    free(pred_prob)

    # 8. Rescale to base peak = 1.0
    cdef double mx = 0.0
    for i in range(<size_t>n_combined):
        if combined_int[i] > mx:
            mx = combined_int[i]
    if mx > 0.0:
        for i in range(<size_t>n_combined):
            combined_int[i] /= mx

    # 9. Match observed peaks to closest predicted
    cdef double best_diff, d_val, pred_val, rmse, match_frac
    cdef int best_j, n_matched = 0
    cdef double sse = 0.0
    cdef int count = 0, base_idx = 0
    cdef double max_obs = obs_int[0]
    cdef bint matched

    # Find base peak index
    for i in range(1, <size_t>n_obs):
        if obs_int[i] > max_obs:
            max_obs = obs_int[i]
            base_idx = <int>i

    # Match and score
    for i in range(<size_t>n_obs):
        best_diff = 1e30
        best_j = -1
        for j in range(<size_t>n_combined):
            d_val = fabs(obs_mz[i] - combined_mz[j])
            if d_val < best_diff:
                best_diff = d_val
                best_j = <int>j

        pred_val = 0.0
        matched = False
        if best_diff <= match_tol:
            pred_val = combined_int[best_j]
            matched = True
            n_matched += 1

        if <int>i != base_idx:
            d_val = obs_int[i] - pred_val
            sse += d_val * d_val
            count += 1

    free(combined_mz)
    free(combined_int)

    rmse = sqrt(sse / count) if count > 0 else 0.0
    match_frac = <double>n_matched / <double>n_obs if n_obs > 0 else 0.0

    return (rmse, match_frac, n_matched)


def score_isotope_batch(
    list symbols,
    np.ndarray counts_2d,
    int charge,
    np.ndarray observed_envelope,
    double mz_match_tolerance,
    double simulated_mz_tolerance = 0.05,
    double simulated_intensity_threshold = 0.001,
):
    """
    Batch isotope envelope scoring for multiple candidates.

    Drop-in replacement for the Numba-based score_isotope_batch.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])
        counts_2d: int32 array of shape (N, n_elements) with atom counts
        charge: Ion charge state
        observed_envelope: 2D array of [m/z, intensity] pairs (normalized)
        mz_match_tolerance: Max m/z difference for peak matching (Da)
        simulated_mz_tolerance: Resolution for combining isotopologues
        simulated_intensity_threshold: Min relative intensity threshold

    Returns:
        Tuple of (rmse_arr, match_frac_arr, n_matched_arr)
    """
    _load_isospec_lib()

    from ._isospec_bridge import get_isotope_arrays
    from molmass.elements import ELECTRON

    cdef int n_elements = len(symbols)
    counts_2d = np.ascontiguousarray(counts_2d, dtype=np.int32)
    cdef int n_candidates = counts_2d.shape[0]

    iso_numbers_np, flat_masses_np, flat_probs_np = get_isotope_arrays(symbols)

    cdef np.ndarray iso_numbers = np.ascontiguousarray(iso_numbers_np, dtype=np.int32)
    cdef np.ndarray flat_masses = np.ascontiguousarray(flat_masses_np, dtype=np.float64)
    cdef np.ndarray flat_probs = np.ascontiguousarray(flat_probs_np, dtype=np.float64)

    cdef np.ndarray obs_mz = np.ascontiguousarray(observed_envelope[:, 0], dtype=np.float64)
    cdef np.ndarray obs_int = np.ascontiguousarray(observed_envelope[:, 1], dtype=np.float64)
    cdef int n_obs = obs_mz.shape[0]

    cdef double electron_mass = ELECTRON.mass

    # Output arrays
    cdef np.ndarray rmse_out = np.empty(n_candidates, dtype=np.float64)
    cdef np.ndarray mf_out = np.empty(n_candidates, dtype=np.float64)
    cdef np.ndarray nm_out = np.empty(n_candidates, dtype=np.int32)

    # Temporary atom_counts array
    cdef np.ndarray atom_counts = np.empty(n_elements, dtype=np.int32)

    cdef int i, j
    cdef double r, mf
    cdef int nm

    # Get raw pointers for nogil block via memoryviews
    cdef int32_t[::1] iso_numbers_view = iso_numbers
    cdef double[::1] flat_masses_view = flat_masses
    cdef double[::1] flat_probs_view = flat_probs
    cdef double[::1] obs_mz_view = obs_mz
    cdef double[::1] obs_int_view = obs_int
    cdef int32_t[::1] atom_counts_view = atom_counts
    cdef int32_t[:, ::1] counts_view = counts_2d
    cdef double[::1] rmse_out_view = rmse_out
    cdef double[::1] mf_out_view = mf_out
    cdef int32_t[::1] nm_out_view = nm_out

    with nogil:
        for i in range(n_candidates):
            for j in range(n_elements):
                atom_counts_view[j] = counts_view[i, j]

            r, mf, nm = _score_single_envelope(
                &iso_numbers_view[0], &atom_counts_view[0],
                &flat_masses_view[0], &flat_probs_view[0], n_elements,
                &obs_mz_view[0], &obs_int_view[0], n_obs,
                simulated_mz_tolerance, mz_match_tolerance,
                simulated_intensity_threshold,
                charge, electron_mass,
            )
            rmse_out_view[i] = r
            mf_out_view[i] = mf
            nm_out_view[i] = nm

    return rmse_out, mf_out, nm_out


def match_isotope_envelope_fast(
    list symbols,
    list counts,
    int charge,
    np.ndarray observed_envelope,
    double mz_match_tolerance,
    double simulated_mz_tolerance = 0.05,
    double simulated_intensity_threshold = 0.001,
):
    """
    Fast isotope envelope matching — single candidate.

    Drop-in replacement for the Numba-based match_isotope_envelope_fast.
    """
    from .results import SingleEnvelopeMatchResult

    # Filter to non-zero elements
    active_symbols = []
    active_counts = []
    for sym, cnt in zip(symbols, counts):
        if cnt > 0:
            active_symbols.append(sym)
            active_counts.append(cnt)

    n_obs = observed_envelope.shape[0]
    if not active_symbols:
        return SingleEnvelopeMatchResult(
            num_peaks_matched=0,
            num_peaks_total=n_obs,
            intensity_rmse=1.0,
            match_fraction=0.0,
            peak_matches=np.full(n_obs, False),
            predicted_envelope=np.empty((0, 2), dtype=np.float64),
        )

    # Build 2D array for batch scorer
    cdef np.ndarray[int32_t, ndim=2] counts_2d = np.array(
        [active_counts], dtype=np.int32,
    )

    rmse_arr, mf_arr, nm_arr = score_isotope_batch(
        active_symbols, counts_2d, charge,
        observed_envelope, mz_match_tolerance,
        simulated_mz_tolerance, simulated_intensity_threshold,
    )

    peak_matches = np.full(n_obs, int(nm_arr[0]) > 0)

    return SingleEnvelopeMatchResult(
        num_peaks_matched=int(nm_arr[0]),
        num_peaks_total=n_obs,
        intensity_rmse=float(rmse_arr[0]),
        match_fraction=float(mf_arr[0]),
        peak_matches=peak_matches,
        predicted_envelope=np.empty((0, 2), dtype=np.float64),
    )
