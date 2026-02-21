"""
Detailed profiling of the IsoSpec isotope scoring pipeline.
Break down: setupIso, setupThreshold, read arrays, combine, match, cleanup.
"""
import time
import numpy as np
from find_mfs import FormulaFinder
from find_mfs.isotopes._isospec_bridge import get_isotope_arrays
from find_mfs.isotopes import SingleEnvelopeMatch


def bench_get_isotope_arrays():
    """Measure get_isotope_arrays overhead."""
    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    n = 10000
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        iso_numbers, flat_masses, flat_probs = get_isotope_arrays(symbols)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"get_isotope_arrays:  median={np.median(arr):.2f} us, mean={np.mean(arr):.2f} us")


def bench_score_isotope_batch():
    """Measure score_isotope_batch for various batch sizes."""
    from find_mfs.isotopes._isospec import score_isotope_batch

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])

    # Single candidate
    counts_1 = np.array([[6, 12, 0, 6, 0, 0]], dtype=np.int32)
    n = 2000
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_1, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"\nscore_isotope_batch (1 candidate):   median={np.median(arr):.1f} us")

    # 5 candidates
    counts_5 = np.array([
        [6, 12, 0, 6, 0, 0],
        [5, 10, 2, 4, 0, 0],
        [4, 8, 0, 8, 0, 0],
        [7, 14, 1, 3, 0, 1],
        [3, 6, 0, 10, 0, 0],
    ], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_5, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"score_isotope_batch (5 candidates):  median={np.median(arr):.1f} us, per-candidate={np.median(arr)/5:.1f} us")

    # 50 candidates
    counts_50 = np.tile(counts_5, (10, 1))
    times = []
    for _ in range(500):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_50, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"score_isotope_batch (50 candidates): median={np.median(arr):.1f} us, per-candidate={np.median(arr)/50:.1f} us")

    # 200 candidates
    counts_200 = np.tile(counts_5, (40, 1))
    times = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_200, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"score_isotope_batch (200 candidates): median={np.median(arr):.1f} us, per-candidate={np.median(arr)/200:.1f} us")


def bench_cython_vs_python_path():
    """Compare Cython batch path vs Python match_isotope_envelope."""
    from find_mfs.isotopes.envelope import match_isotope_envelope
    from find_mfs.core._light_formula import LightFormula

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])

    formula = LightFormula.from_counts(
        symbols=symbols, counts=[6, 12, 0, 6, 0, 0],
        charge=0, monoisotopic_mass=180.063,
    )

    # Python path
    n = 500
    times_py = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        match_isotope_envelope(formula, envelope, mz_match_tolerance=0.01)
        t1 = time.perf_counter_ns()
        times_py.append((t1 - t0) / 1e3)

    # Cython path (single)
    from find_mfs.isotopes._isospec import score_isotope_batch
    counts_1 = np.array([[6, 12, 0, 6, 0, 0]], dtype=np.int32)
    times_cy = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_1, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times_cy.append((t1 - t0) / 1e3)

    py_arr = np.array(times_py)
    cy_arr = np.array(times_cy)
    print(f"\nPython match_isotope_envelope: median={np.median(py_arr):.1f} us")
    print(f"Cython score_isotope_batch:   median={np.median(cy_arr):.1f} us")
    print(f"Speedup: {np.median(py_arr)/np.median(cy_arr):.1f}x")


def bench_full_pipeline_with_isotope():
    """Measure end-to-end find_formulae with isotope, varying result counts."""
    finder = FormulaFinder('CHNOPS')

    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])
    iso_config = SingleEnvelopeMatch(envelope, mz_tolerance_da=0.01)

    # Mass 180 (few results before iso filter)
    n = 1000
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        r = finder.find_formulae(mass=180.063388, error_ppm=5.0, isotope_match=iso_config)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"\nfind_formulae + isotope (mass=180, {len(r)} results): median={np.median(arr):.1f} us")

    # Mass 500 (more results)
    envelope_500 = np.array([
        [500.0, 1.00],
        [501.003, 0.22],
        [502.006, 0.035],
    ])
    iso_500 = SingleEnvelopeMatch(envelope_500, mz_tolerance_da=0.01)
    times = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        r = finder.find_formulae(mass=500.0, error_ppm=5.0, isotope_match=iso_500)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"find_formulae + isotope (mass=500, {len(r)} results): median={np.median(arr):.1f} us")

    # How many candidates go into isotope scoring? (before iso filter)
    r_no_iso = finder.find_formulae(mass=500.0, error_ppm=5.0)
    print(f"  (candidates before iso filter: {len(r_no_iso)})")

    # With RDBE + isotope (fewer candidates to score)
    times = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        r = finder.find_formulae(
            mass=500.0, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
            isotope_match=iso_500,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    r_rdbe = finder.find_formulae(mass=500.0, charge=1, error_ppm=5.0,
                                   filter_rdbe=(-0.5, 40), check_octet=True)
    print(f"find_formulae + RDBE + isotope (mass=500, {len(r)} results): median={np.median(arr):.1f} us")
    print(f"  (candidates after RDBE, before iso: {len(r_rdbe)})")


def bench_setupIso_overhead():
    """Try to isolate setupIso vs the rest by timing with different element counts."""
    from find_mfs.isotopes._isospec import score_isotope_batch

    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
    ])

    # 2 elements (CH)
    counts = np.array([[6, 12]], dtype=np.int32)
    n = 3000
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(['C', 'H'], counts, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"\n2 elements (CH):     median={np.median(arr):.1f} us")

    # 4 elements (CHNO)
    counts = np.array([[6, 12, 0, 6]], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(['C', 'H', 'N', 'O'], counts, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"4 elements (CHNO):   median={np.median(arr):.1f} us")

    # 6 elements (CHNOPS)
    counts = np.array([[6, 12, 0, 6, 0, 0]], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(['C', 'H', 'N', 'O', 'P', 'S'], counts, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"6 elements (CHNOPS): median={np.median(arr):.1f} us")

    # 10 elements (CHNOPS+halogens)
    counts = np.array([[6, 12, 0, 6, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(
            ['C', 'H', 'N', 'O', 'P', 'S', 'Cl', 'Br', 'F', 'I'],
            counts, 0, envelope, 0.01,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"10 elements (+halo): median={np.median(arr):.1f} us")

    # Same 6 elements but larger molecule
    counts = np.array([[30, 50, 5, 10, 1, 2]], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(['C', 'H', 'N', 'O', 'P', 'S'], counts, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"6 elem, large mol:   median={np.median(arr):.1f} us")

    # Very large molecule
    counts = np.array([[60, 100, 10, 20, 2, 4]], dtype=np.int32)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(['C', 'H', 'N', 'O', 'P', 'S'], counts, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"6 elem, very large:  median={np.median(arr):.1f} us")


def bench_python_wrapper_overhead():
    """Measure overhead of score_isotope_batch Python wrapper vs pure C."""
    from find_mfs.isotopes._isospec import score_isotope_batch, _load_isospec_lib
    _load_isospec_lib()

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    counts_2d = np.array([[6, 12, 0, 6, 0, 0]] * 10, dtype=np.int32)
    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])

    n = 1000
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        score_isotope_batch(symbols, counts_2d, 0, envelope, 0.01)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    arr = np.array(times)
    print(f"\nscore_isotope_batch (10 candidates, full): median={np.median(arr):.1f} us, per-cand={np.median(arr)/10:.1f} us")


if __name__ == '__main__':
    print("="*60)
    print("ISOSPEC PROFILING")
    print("="*60)

    bench_get_isotope_arrays()
    bench_score_isotope_batch()
    bench_setupIso_overhead()
    bench_cython_vs_python_path()
    bench_python_wrapper_overhead()
    bench_full_pipeline_with_isotope()
