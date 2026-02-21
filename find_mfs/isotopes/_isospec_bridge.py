"""
Low-level ctypes bridge to IsoSpecPy's C++ shared library.

Provides:
- ctypes function references callable from Numba @njit functions
- Per-element isotope data arrays from IsoSpecPy.PeriodicTbl
- uint64_to_voidptr intrinsic for pointer casting in Numba

This module is loaded lazily; import errors are deferred until first use.
"""
from __future__ import annotations

import ctypes
from ctypes import c_int, c_double, c_size_t, c_void_p, c_bool, POINTER

import numpy as np
from numba import types as nbtypes
from numba.core import cgutils
from numba.extending import intrinsic

# ---------------------------------------------------------------------------
# Numba intrinsic: cast uint64 (ctypes c_void_p return) to voidptr for carray
# ---------------------------------------------------------------------------

@intrinsic
def uint64_to_voidptr(typingctx, src):
    """Cast a uint64 (from ctypes c_void_p return) to Numba voidptr."""
    sig = nbtypes.voidptr(nbtypes.uint64)
    def codegen(context, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


# ---------------------------------------------------------------------------
# Load IsoSpecPy's C++ shared library via ctypes
# ---------------------------------------------------------------------------

_lib = None
_lib_loaded = False


def _ensure_lib():
    """Load the C++ library on first use."""
    global _lib, _lib_loaded
    if _lib_loaded:
        return
    try:
        from IsoSpecPy.isoFFI import isoFFI
        _lib = ctypes.CDLL(str(isoFFI.libpath))
    except (ImportError, OSError) as e:
        raise ImportError(
            f"Cannot load IsoSpecPy C++ library: {e}. "
            "Install IsoSpecPy with: pip install IsoSpecPy"
        ) from e
    _setup_signatures()
    _lib_loaded = True


def _setup_signatures():
    """Define C function signatures on the loaded library."""
    lib = _lib

    # void* setupIso(int dimNumber, const int* isotopeNumbers,
    #                const int* atomCounts, const double* isotopeMasses,
    #                const double* isotopeProbabilities)
    lib.setupIso.restype = c_void_p
    lib.setupIso.argtypes = [
        c_int,
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_double),
    ]

    # void* setupThresholdFixedEnvelope(void* iso, double threshold,
    #                                    bool absolute, bool get_confs)
    lib.setupThresholdFixedEnvelope.restype = c_void_p
    lib.setupThresholdFixedEnvelope.argtypes = [
        c_void_p, c_double, c_bool, c_bool,
    ]

    # size_t confs_noFixedEnvelope(void* tabulator)
    lib.confs_noFixedEnvelope.restype = c_size_t
    lib.confs_noFixedEnvelope.argtypes = [c_void_p]

    # const double* massesFixedEnvelope(void* tabulator)
    lib.massesFixedEnvelope.restype = c_void_p
    lib.massesFixedEnvelope.argtypes = [c_void_p]

    # const double* probsFixedEnvelope(void* tabulator)
    lib.probsFixedEnvelope.restype = c_void_p
    lib.probsFixedEnvelope.argtypes = [c_void_p]

    # void deleteFixedEnvelope(void* tabulator, bool releaseEverything)
    lib.deleteFixedEnvelope.restype = None
    lib.deleteFixedEnvelope.argtypes = [c_void_p, c_bool]

    # void deleteIso(void* iso)
    lib.deleteIso.restype = None
    lib.deleteIso.argtypes = [c_void_p]

    # void freeReleasedArray(void* array)
    lib.freeReleasedArray.restype = None
    lib.freeReleasedArray.argtypes = [c_void_p]


# ---------------------------------------------------------------------------
# Public ctypes function references (for use from Numba @njit)
# ---------------------------------------------------------------------------

def get_lib_functions():
    """
    Return ctypes function references for use in Numba @njit code.

    Returns tuple of:
        (setupIso, setupThresholdFixedEnvelope, confs_noFixedEnvelope,
         massesFixedEnvelope, probsFixedEnvelope, deleteFixedEnvelope,
         deleteIso, freeReleasedArray)
    """
    _ensure_lib()
    return (
        _lib.setupIso,
        _lib.setupThresholdFixedEnvelope,
        _lib.confs_noFixedEnvelope,
        _lib.massesFixedEnvelope,
        _lib.probsFixedEnvelope,
        _lib.deleteFixedEnvelope,
        _lib.deleteIso,
        _lib.freeReleasedArray,
    )


# ---------------------------------------------------------------------------
# Isotope data arrays from IsoSpecPy.PeriodicTbl
# ---------------------------------------------------------------------------

_isotope_cache: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}


def _build_isotope_cache():
    """Build per-element isotope data from IsoSpecPy's PeriodicTbl."""
    if _isotope_cache:
        return
    try:
        from IsoSpecPy import PeriodicTbl
    except ImportError as e:
        raise ImportError(
            "IsoSpecPy is required for isotope data. "
            "Install with: pip install IsoSpecPy"
        ) from e

    for symbol in PeriodicTbl.symbol_to_masses:
        masses = np.array(PeriodicTbl.symbol_to_masses[symbol], dtype=np.float64)
        probs = np.array(PeriodicTbl.symbol_to_probs[symbol], dtype=np.float64)
        _isotope_cache[symbol] = (len(masses), masses, probs)


def get_isotope_arrays(
    symbols: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat isotope data arrays for setupIso from element symbols.

    Args:
        symbols: Element symbols (e.g., ['C', 'H', 'N', 'O', 'P', 'S'])

    Returns:
        Tuple of:
        - iso_numbers: int32 array of isotope counts per element
        - flat_masses: float64 array of all isotope masses (concatenated)
        - flat_probs: float64 array of all isotope probabilities (concatenated)
    """
    _build_isotope_cache()

    iso_numbers = np.empty(len(symbols), dtype=np.int32)
    all_masses = []
    all_probs = []

    for i, sym in enumerate(symbols):
        n, masses, probs = _isotope_cache[sym]
        iso_numbers[i] = n
        all_masses.append(masses)
        all_probs.append(probs)

    flat_masses = np.concatenate(all_masses)
    flat_probs = np.concatenate(all_probs)
    return iso_numbers, flat_masses, flat_probs


# ---------------------------------------------------------------------------
# M+1 / M+2 approximation coefficients for pre-filter
# ---------------------------------------------------------------------------

# M+1 ratio: probability of a single atom contributing a +1 neutron isotope
# relative to the monoisotopic peak. These are P(M+1)/P(M) per atom.
M1_RATIOS: dict[str, float] = {
    'C': 0.010816, 'H': 0.000115, 'N': 0.003654,
    'O': 0.000381, 'P': 0.0, 'S': 0.007895,
    'F': 0.0, 'Cl': 0.003200, 'Br': 0.009700, 'I': 0.0,
    'Si': 0.050800, 'Se': 0.0,
}

# M+2 "direct" ratio: contribution of a single atom to M+2 that is NOT
# accounted for by (M+1)^2/2. This captures elements with significant
# M+2 isotopes (like Cl-37, S-34, etc.).
M2_DIRECT: dict[str, float] = {
    'C': 0.0, 'H': 0.0, 'N': 0.0,
    'O': 0.002055, 'P': 0.0, 'S': 0.044742,
    'F': 0.0, 'Cl': 0.320300, 'Br': 0.970300, 'I': 0.0,
    'Si': 0.033600, 'Se': 0.0,
}
