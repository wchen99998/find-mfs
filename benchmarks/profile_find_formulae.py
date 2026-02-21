"""
Profile FormulaFinder.find_formulae() to identify Python overhead.

Measures time spent in each phase:
1. Argument parsing / adduct / constraint prep  (Python)
2. decompose_and_score (mostly Cython)
3. Isotope batch scoring (Cython + numpy)
4. Result construction (Python objects)
"""
import time
import numpy as np
import cProfile
import pstats
import io

from find_mfs import FormulaFinder
from find_mfs.isotopes import SingleEnvelopeMatch


def make_finder(elements='CHNOPS'):
    return FormulaFinder(elements)


def bench_simple(finder, n=1000):
    """Simple search — no RDBE, no octet, no isotope."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=180.063388,
            error_ppm=5.0,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # us
    return times


def bench_with_rdbe(finder, n=1000):
    """With RDBE + octet filtering (pushed into Cython)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=180.063388,
            charge=1,
            error_ppm=5.0,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return times


def bench_with_adduct(finder, n=1000):
    """With Na adduct."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=203.05261,
            charge=1,
            adduct="Na",
            error_ppm=5.0,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return times


def bench_with_isotope(finder, n=200):
    """With isotope matching (full pipeline)."""
    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])
    iso_config = SingleEnvelopeMatch(envelope, mz_tolerance_da=0.01)
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=180.063388,
            error_ppm=5.0,
            isotope_match=iso_config,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return times


def bench_large_mass(finder, n=200):
    """Large mass = more candidates, more Python object overhead."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=800.0,
            error_ppm=5.0,
        )
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
        n_results = len(result)
    return times, n_results


def bench_large_mass_with_access(finder, n=200):
    """Large mass + iterate all results (forces materialization)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=800.0,
            error_ppm=5.0,
        )
        # Force materialization of all
        for r in result:
            _ = r.formula
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
        n_results = len(result)
    return times, n_results


def report(name, times_us):
    arr = np.array(times_us)
    print(f"\n{name}:")
    print(f"  n={len(arr)}, median={np.median(arr):.1f} us, "
          f"mean={np.mean(arr):.1f} us, p95={np.percentile(arr, 95):.1f} us, "
          f"min={np.min(arr):.1f} us")


def cprofile_run(finder):
    """Run cProfile on a typical call to see function-level breakdown."""
    envelope = np.array([
        [180.063388, 1.00],
        [181.066743, 0.065],
        [182.068, 0.012],
    ])
    iso_config = SingleEnvelopeMatch(envelope, mz_tolerance_da=0.01)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(500):
        finder.find_formulae(
            mass=180.063388,
            error_ppm=5.0,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print("\n=== cProfile (500 calls, simple+RDBE) ===")
    print(s.getvalue())

    pr2 = cProfile.Profile()
    pr2.enable()
    for _ in range(200):
        finder.find_formulae(
            mass=180.063388,
            error_ppm=5.0,
            isotope_match=iso_config,
        )
    pr2.disable()

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2)
    ps2.sort_stats('cumulative')
    ps2.print_stats(30)
    print("\n=== cProfile (200 calls, with isotope) ===")
    print(s2.getvalue())

    # Large mass
    pr3 = cProfile.Profile()
    pr3.enable()
    for _ in range(50):
        r = finder.find_formulae(mass=800.0, error_ppm=5.0)
        for x in r:
            _ = x.formula
    pr3.disable()

    s3 = io.StringIO()
    ps3 = pstats.Stats(pr3, stream=s3)
    ps3.sort_stats('cumulative')
    ps3.print_stats(30)
    print(f"\n=== cProfile (50 calls, mass=800, full materialization, {len(r)} results) ===")
    print(s3.getvalue())


def phase_breakdown(finder):
    """Manually instrument find_formulae phases."""
    import types
    from find_mfs.core.decomposer import MassDecomposer

    # Monkey-patch to time individual phases
    original_decompose = finder.decomposer.decompose_and_score
    decompose_times = []

    def timed_decompose(self_dec, *args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original_decompose(*args, **kwargs)
        t1 = time.perf_counter_ns()
        decompose_times.append((t1 - t0) / 1e3)
        return result

    finder.decomposer.decompose_and_score = types.MethodType(timed_decompose, finder.decomposer)

    n = 500
    total_times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        result = finder.find_formulae(
            mass=180.063388,
            error_ppm=5.0,
            filter_rdbe=(-0.5, 40),
            check_octet=True,
        )
        t1 = time.perf_counter_ns()
        total_times.append((t1 - t0) / 1e3)

    total_arr = np.array(total_times)
    decompose_arr = np.array(decompose_times)
    overhead_arr = total_arr - decompose_arr

    print(f"\n=== Phase breakdown (n={n}, mass=180, RDBE+octet) ===")
    print(f"  Total:      median={np.median(total_arr):.1f} us, mean={np.mean(total_arr):.1f} us")
    print(f"  Decompose:  median={np.median(decompose_arr):.1f} us, mean={np.mean(decompose_arr):.1f} us")
    print(f"  Python OH:  median={np.median(overhead_arr):.1f} us, mean={np.mean(overhead_arr):.1f} us")
    print(f"  OH fraction: {np.mean(overhead_arr)/np.mean(total_arr)*100:.1f}%")

    # Restore
    finder.decomposer.decompose_and_score = original_decompose


if __name__ == '__main__':
    print("Building FormulaFinder...")
    finder = make_finder('CHNOPS')

    # Warmup
    finder.find_formulae(mass=180.063388, error_ppm=5.0)

    print("\n" + "="*60)
    print("MICROBENCHMARKS")
    print("="*60)

    report("Simple (no filters)", bench_simple(finder))
    report("RDBE + octet", bench_with_rdbe(finder))
    report("Na adduct", bench_with_adduct(finder))
    report("Isotope match", bench_with_isotope(finder))

    t_large, n_large = bench_large_mass(finder, n=100)
    report(f"Large mass (800 Da, {n_large} results, lazy)", t_large)

    t_large_eager, n_large2 = bench_large_mass_with_access(finder, n=50)
    report(f"Large mass (800 Da, {n_large2} results, full materialization)", t_large_eager)

    print("\n" + "="*60)
    print("PHASE BREAKDOWN")
    print("="*60)
    phase_breakdown(finder)

    print("\n" + "="*60)
    print("CPROFILE DETAILED")
    print("="*60)
    cprofile_run(finder)
