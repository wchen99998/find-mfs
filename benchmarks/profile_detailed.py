"""
Detailed microbenchmark: break down Python overhead per-phase.
"""
import time
import numpy as np
from find_mfs import FormulaFinder
from find_mfs.core._light_formula import LightFormula
from find_mfs.core.finder import FormulaCandidate
from find_mfs.core.results import _LazyBackend, FormulaSearchResults


def bench_decomposer_overhead():
    """Measure decomposer.decompose_and_score overhead vs raw Cython."""
    finder = FormulaFinder('CHNOPS')
    decomposer = finder.decomposer

    # Time the full decompose_and_score (Python wrapper + Cython)
    from find_mfs.core.algorithms import decompose_and_score as _cython_das

    # Prepare args manually (what decompose_and_score does internally)
    max_counts = {s: 10000 for s in decomposer.element_symbols}
    min_counts = {s: 0 for s in decomposer.element_symbols}
    bounds, min_values = decomposer._get_possible_element_count_bounds(max_counts, min_counts)

    adjusted_mass = 180.063388
    charge = 0
    ppm_error = 5.0
    charge_mass_offset = 0.0

    original_min_mass, original_max_mass = decomposer._populate_mass_range_based_on_user_input(
        mass=adjusted_mass, ppm_error=ppm_error, mz_error=0.0,
    )
    min_mass, max_mass = decomposer._populate_mass_range_based_on_user_input(
        mass=adjusted_mass, ppm_error=ppm_error, mz_error=0.0,
        min_counts=min_counts,
    )

    import math
    for i, element in enumerate(decomposer.elements):
        mass_bound = math.floor(max_mass / element.mass)
        bounds[i] = min(bounds[i], float(mass_bound))

    min_int, max_int = decomposer._get_mass_range_as_integers(
        min_mass=max(0.0, min_mass), max_mass=max(0.0, max_mass),
    )

    integer_masses = decomposer.integer_masses
    real_masses = decomposer.real_masses
    bounds_arr = np.array(bounds, dtype=np.float64)
    min_values_arr = np.array(min_values, dtype=np.int64)

    rdbe_coeffs_arr = np.zeros(len(decomposer.elements), dtype=np.float64)
    iso_m1_arr = np.zeros(len(decomposer.elements), dtype=np.float64)
    iso_m2_arr = np.zeros(len(decomposer.elements), dtype=np.float64)

    # Bench raw Cython call
    n = 2000
    times_raw = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        _cython_das(
            decomposer.ERT, integer_masses, real_masses,
            bounds_arr, min_values_arr,
            min_int, max_int,
            original_min_mass, original_max_mass,
            charge_mass_offset, 10000,
            rdbe_coeffs_arr, -np.inf, np.inf,
            False, True, False,
            False, iso_m1_arr, iso_m2_arr,
            0.0, 0.0, 0.3, 0.02,
            180.063388, 0.0, False,
        )
        t1 = time.perf_counter_ns()
        times_raw.append((t1 - t0) / 1e3)

    # Bench Python wrapper
    times_wrapper = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        decomposer.decompose_and_score(
            query_mass=180.063388, charge=0,
            ppm_error=5.0, mz_error=0.0,
            max_results=10000,
            ion_query_mass=180.063388, adduct_mass=0.0,
        )
        t1 = time.perf_counter_ns()
        times_wrapper.append((t1 - t0) / 1e3)

    raw_arr = np.array(times_raw)
    wrap_arr = np.array(times_wrapper)
    print(f"\n=== Decomposer overhead ===")
    print(f"  Raw Cython:    median={np.median(raw_arr):.1f} us")
    print(f"  Python wrapper: median={np.median(wrap_arr):.1f} us")
    print(f"  Wrapper OH:    {np.median(wrap_arr) - np.median(raw_arr):.1f} us "
          f"({(np.median(wrap_arr) - np.median(raw_arr))/np.median(wrap_arr)*100:.1f}%)")


def bench_materialization():
    """Measure LightFormula.from_counts + FormulaCandidate construction."""
    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    counts = [6, 12, 0, 6, 0, 0]
    n = 100000

    # LightFormula.from_counts
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        f = LightFormula.from_counts(symbols=symbols, counts=counts, charge=0, monoisotopic_mass=180.063)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)

    arr = np.array(times)
    print(f"\n=== LightFormula.from_counts ===")
    print(f"  median={np.median(arr):.3f} us, mean={np.mean(arr):.3f} us")

    # FormulaCandidate construction
    times2 = []
    for _ in range(n):
        f = LightFormula.from_counts(symbols=symbols, counts=counts, charge=0, monoisotopic_mass=180.063)
        t0 = time.perf_counter_ns()
        fc = FormulaCandidate(formula=f, error_ppm=1.5, error_da=0.0003, rdbe=5.0, adduct=None)
        t1 = time.perf_counter_ns()
        times2.append((t1 - t0) / 1e3)

    arr2 = np.array(times2)
    print(f"\n=== FormulaCandidate dataclass ===")
    print(f"  median={np.median(arr2):.3f} us, mean={np.mean(arr2):.3f} us")

    # Full _materialize simulation (what _LazyBackend does)
    # Build a mock raw dict
    counts_arr = np.tile(np.array([6, 12, 0, 6, 0, 0], dtype=np.int32), (1000, 1))
    raw = {
        'counts': counts_arr,
        'exact_masses': np.full(1000, 180.063, dtype=np.float64),
        'error_ppm': np.full(1000, 1.5, dtype=np.float64),
        'error_da': np.full(1000, 0.0003, dtype=np.float64),
        'rdbe': np.full(1000, 5.0, dtype=np.float64),
    }
    backend = _LazyBackend(raw=raw, symbols=symbols, charge=0)

    times3 = []
    for i in range(1000):
        backend._cache.clear()
        t0 = time.perf_counter_ns()
        _ = backend._materialize(i)
        t1 = time.perf_counter_ns()
        times3.append((t1 - t0) / 1e3)

    arr3 = np.array(times3)
    print(f"\n=== _LazyBackend._materialize (per item) ===")
    print(f"  median={np.median(arr3):.3f} us, mean={np.mean(arr3):.3f} us")

    # ndarray.tolist() overhead
    row = counts_arr[0]
    times4 = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        _ = row.tolist()
        t1 = time.perf_counter_ns()
        times4.append((t1 - t0) / 1e3)

    arr4 = np.array(times4)
    print(f"\n=== ndarray.tolist() (1 row, 6 elements) ===")
    print(f"  median={np.median(arr4):.3f} us, mean={np.mean(arr4):.3f} us")


def bench_find_formulae_python_overhead():
    """Measure just the Python code in find_formulae (excluding Cython)."""
    finder = FormulaFinder('CHNOPS')

    # Time the full find_formulae vs just decompose_and_score
    n = 2000

    # Direct Cython call with pre-prepared args
    from find_mfs.core.algorithms import decompose_and_score as _cython_das
    decomposer = finder.decomposer
    import math

    max_counts = {s: 10000 for s in decomposer.element_symbols}
    min_counts = {s: 0 for s in decomposer.element_symbols}
    bounds, min_values = decomposer._get_possible_element_count_bounds(max_counts, min_counts)
    adjusted_mass = 180.063388
    original_min_mass, original_max_mass = decomposer._populate_mass_range_based_on_user_input(
        mass=adjusted_mass, ppm_error=5.0, mz_error=0.0,
    )
    min_mass, max_mass = decomposer._populate_mass_range_based_on_user_input(
        mass=adjusted_mass, ppm_error=5.0, mz_error=0.0, min_counts=min_counts,
    )
    for i, element in enumerate(decomposer.elements):
        mass_bound = math.floor(max_mass / element.mass)
        bounds[i] = min(bounds[i], float(mass_bound))
    min_int, max_int = decomposer._get_mass_range_as_integers(
        min_mass=max(0.0, min_mass), max_mass=max(0.0, max_mass),
    )
    bounds_arr = np.array(bounds, dtype=np.float64)
    min_values_arr = np.array(min_values, dtype=np.int64)
    rdbe_coeffs_arr = finder._rdbe_coeffs
    iso_m1_arr = np.zeros(6, dtype=np.float64)
    iso_m2_arr = np.zeros(6, dtype=np.float64)

    times_cython = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        raw = _cython_das(
            decomposer.ERT, decomposer.integer_masses, decomposer.real_masses,
            bounds_arr, min_values_arr,
            min_int, max_int,
            original_min_mass, original_max_mass,
            0.0, 10000,
            rdbe_coeffs_arr, -0.5, 40.0,
            True, True, True,
            False, iso_m1_arr, iso_m2_arr,
            0.0, 0.0, 0.3, 0.02,
            180.063388, 0.0, True,
        )
        t1 = time.perf_counter_ns()
        times_cython.append((t1 - t0) / 1e3)

    times_full = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=180.063388, charge=1,
            error_ppm=5.0, filter_rdbe=(-0.5, 40), check_octet=True,
        )
        t1 = time.perf_counter_ns()
        times_full.append((t1 - t0) / 1e3)

    cython_arr = np.array(times_cython)
    full_arr = np.array(times_full)
    oh = np.median(full_arr) - np.median(cython_arr)
    print(f"\n=== find_formulae Python overhead (RDBE+octet, mass=180) ===")
    print(f"  Raw Cython decompose+score: median={np.median(cython_arr):.1f} us")
    print(f"  Full find_formulae:         median={np.median(full_arr):.1f} us")
    print(f"  Python overhead:            {oh:.1f} us ({oh/np.median(full_arr)*100:.1f}%)")


if __name__ == '__main__':
    print("="*60)
    print("DETAILED MICROBENCHMARKS")
    print("="*60)
    bench_decomposer_overhead()
    bench_materialization()
    bench_find_formulae_python_overhead()
