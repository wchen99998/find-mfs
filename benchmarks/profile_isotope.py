#!/usr/bin/env python3
"""
Profile per-call breakdown of old vs new isotope matching paths.

Reports:
1. Old path (match_isotope_envelope) timing
2. New path (match_isotope_envelope_fast) timing
3. Breakdown of new path components
4. Non-isotope code path regression check
"""

import time
import numpy as np
from molmass import Formula

from find_mfs import FormulaFinder
from find_mfs.isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
    match_isotope_envelope_fast,
)
from find_mfs.isotopes.config import SingleEnvelopeMatch


def time_fn(fn, n_calls=100, warmup=5):
    """Time a function over n_calls, returning median time per call."""
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean, min(times), max(times)


def profile_isotope_matching():
    """Profile old vs new isotope matching."""
    print("=" * 70)
    print("ISOTOPE MATCHING PER-CALL PROFILE")
    print("=" * 70)

    # Test formulae at different sizes
    test_cases = [
        ("C6H12O6", "glucose (180 Da)"),
        ("C31H36N2O11", "novobiocin (612 Da)"),
        ("C66H75Cl2N9O24", "vancomycin (1449 Da)"),
    ]

    for formula_str, desc in test_cases:
        print(f"\n--- {desc} ---")
        f = Formula(formula_str)
        envelope = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)

        # Parse symbols/counts for new path
        comp = f.composition()
        symbols = []
        counts = []
        for element, item in sorted(comp.items()):
            symbols.append(element)
            counts.append(item.count)

        # Old path
        def old_call():
            return match_isotope_envelope(
                formula=f,
                observed_envelope=envelope,
                mz_match_tolerance=0.01,
                simulated_envelope_mz_tolerance=0.05,
                simulated_envelope_intsy_threshold=0.001,
            )

        # New path
        def new_call():
            return match_isotope_envelope_fast(
                symbols=symbols,
                counts=counts,
                charge=0,
                observed_envelope=envelope,
                mz_match_tolerance=0.01,
                simulated_mz_tolerance=0.05,
                simulated_intensity_threshold=0.001,
            )

        n_calls = 200

        old_med, old_mean, old_min, old_max = time_fn(old_call, n_calls)
        new_med, new_mean, new_min, new_max = time_fn(new_call, n_calls)

        speedup = old_med / new_med if new_med > 0 else float('inf')

        print(f"  Old path:  median={old_med*1e6:.0f}µs  mean={old_mean*1e6:.0f}µs  "
              f"min={old_min*1e6:.0f}µs  max={old_max*1e6:.0f}µs")
        print(f"  New path:  median={new_med*1e6:.0f}µs  mean={new_mean*1e6:.0f}µs  "
              f"min={new_min*1e6:.0f}µs  max={new_max*1e6:.0f}µs")
        print(f"  Speedup:   {speedup:.1f}x")

        # Verify results match
        old_r = old_call()
        new_r = new_call()
        print(f"  Old RMSE: {old_r.intensity_rmse:.6f}  New RMSE: {new_r.intensity_rmse:.6f}")


def profile_full_pipeline():
    """Profile full find_formulae with and without isotope matching."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE PROFILE")
    print("=" * 70)

    finder = FormulaFinder("CHNOPS")
    mass = 612.211  # novobiocin-ish

    # Without isotope
    def no_iso():
        return finder.find_formulae(
            mass=mass, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )

    # With isotope
    envelope = np.array([
        [612.211, 1.0],
        [613.215, 0.34],
        [614.218, 0.06],
    ])

    def with_iso_prefilter():
        iso_config = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=True,
        )
        return finder.find_formulae(
            mass=mass, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
            isotope_match=iso_config,
        )

    def with_iso_no_prefilter():
        iso_config = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=False,
        )
        return finder.find_formulae(
            mass=mass, charge=1, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
            isotope_match=iso_config,
        )

    n_calls = 50

    print(f"\n--- 612 Da, charge=1, CHNOPS ---")

    med, mean, mn, mx = time_fn(no_iso, n_calls, warmup=3)
    r = no_iso()
    print(f"  No isotope:        median={med*1e3:.1f}ms  ({len(r)} candidates)")

    med_pre, mean_pre, _, _ = time_fn(with_iso_prefilter, n_calls, warmup=3)
    r_pre = with_iso_prefilter()
    print(f"  With prefilter:    median={med_pre*1e3:.1f}ms  ({len(r_pre)} candidates)")

    med_nopre, mean_nopre, _, _ = time_fn(with_iso_no_prefilter, n_calls, warmup=3)
    r_nopre = with_iso_no_prefilter()
    print(f"  Without prefilter: median={med_nopre*1e3:.1f}ms  ({len(r_nopre)} candidates)")

    if med_nopre > 0:
        print(f"\n  Prefilter speedup over no-prefilter: {med_nopre/med_pre:.1f}x")
    print(f"  Isotope overhead vs no-isotope: {med_pre/med:.1f}x")


def main():
    print("Isotope Matching Performance Profile")
    print("=====================================\n")

    profile_isotope_matching()
    profile_full_pipeline()


if __name__ == "__main__":
    main()
