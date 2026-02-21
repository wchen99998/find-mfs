"""
Advanced microbenchmarks for Numba decomposition optimizations.

Tests:
  1. Branchless RDBE+octet via XOR arithmetic
  2. Speculative result write (write during computation, skip separate loop)
  3. Return exact_mass/rdbe from Numba (avoid numpy recomputation)
  4. Branchless mass range check
  5. Full decomposition with real ERT data — all variants
"""
import os
os.environ['NUMBA_DISABLE_JIT_CACHE'] = '1'

import time
import statistics
import numpy as np
from numba import njit


# ========================================================================
# 1. BRANCHLESS RDBE+OCTET: XOR arithmetic vs branchy
# ========================================================================

@njit(fastmath=True)
def leaf_branchy(c_arr, min_values, real_masses, rdbe_coeffs,
                 charge_mass_offset, orig_min, orig_max,
                 rdbe_min, rdbe_max, check_octet, charge_parity_even,
                 do_rdbe_filter):
    """Current: multiple if/elif branches for RDBE and octet."""
    n = len(real_masses)
    count = 0
    for row in range(len(c_arr)):
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]

        if total > 0 and orig_min <= exact_mass <= orig_max:
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
                count += 1
    return count


@njit(fastmath=True)
def leaf_branchless(c_arr, min_values, real_masses, rdbe_coeffs,
                    charge_mass_offset, orig_min, orig_max,
                    rdbe_min, rdbe_max, check_octet, charge_parity_even,
                    do_rdbe_filter):
    """Branchless: use arithmetic masks for all filter checks."""
    n = len(real_masses)
    count = 0
    # Precompute octet XOR mask (constant across all candidates)
    charge_parity_int = np.int64(charge_parity_even)
    for row in range(len(c_arr)):
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]

        # Branchless combined check:
        # ok = (total > 0) & (exact_mass >= orig_min) & (exact_mass <= orig_max)
        ok = np.int64(total > 0) & np.int64(exact_mass >= orig_min) & np.int64(exact_mass <= orig_max)

        if do_rdbe_filter:
            ok &= np.int64(rdbe >= rdbe_min) & np.int64(rdbe <= rdbe_max)
            if check_octet:
                # XOR trick: octet_ok = charge_parity_even XOR is_half_int
                # even+integer → ok, even+half → bad
                # odd+half → ok, odd+integer → bad
                is_half_int = np.int64(2.0 * rdbe) & np.int64(1)
                # XOR: if charge_parity matches is_half_int → bad (XOR=0)
                #       if they differ → good (XOR=1)
                # Wait: charge_parity_even=1 means even charge.
                # even charge → RDBE must be integer (not half) → is_half_int=0
                # So good = charge_parity_even XOR is_half_int = 1 XOR 0 = 1
                # even charge + half_int: 1 XOR 1 = 0 → bad ✓
                # odd charge + integer: 0 XOR 0 = 0 → bad ✓
                # odd charge + half: 0 XOR 1 = 1 → good ✓
                octet_ok = charge_parity_int ^ is_half_int
                ok &= octet_ok

        count += ok
    return count


# ========================================================================
# 2. SPECULATIVE WRITE: write result during computation, avoid 2nd loop
# ========================================================================

@njit(fastmath=True)
def decomp_separate_write(c_arr, min_values, real_masses, rdbe_coeffs,
                          charge_mass_offset, orig_min, orig_max,
                          rdbe_min, rdbe_max, charge_parity_even,
                          do_rdbe_filter, max_results):
    """Current: compute, check, then separate loop to write result."""
    n = len(real_masses)
    result = np.empty((max_results, n), dtype=np.int32)
    count = 0
    charge_parity_int = np.int64(charge_parity_even)
    for row in range(len(c_arr)):
        if count >= max_results:
            break
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]

        ok = np.int64(total > 0) & np.int64(exact_mass >= orig_min) & np.int64(exact_mass <= orig_max)
        if do_rdbe_filter:
            ok &= np.int64(rdbe >= rdbe_min) & np.int64(rdbe <= rdbe_max)
            is_half_int = np.int64(2.0 * rdbe) & np.int64(1)
            ok &= charge_parity_int ^ is_half_int

        if ok:
            for j in range(n):
                result[count, j] = np.int32(c_arr[row, j] + min_values[j])
            count += 1
    return result[:count]


@njit(fastmath=True)
def decomp_speculative_write(c_arr, min_values, real_masses, rdbe_coeffs,
                             charge_mass_offset, orig_min, orig_max,
                             rdbe_min, rdbe_max, charge_parity_even,
                             do_rdbe_filter, max_results):
    """Speculative: write result during computation, only advance count if ok."""
    n = len(real_masses)
    result = np.empty((max_results, n), dtype=np.int32)
    count = 0
    charge_parity_int = np.int64(charge_parity_even)
    for row in range(len(c_arr)):
        if count >= max_results:
            break
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]
            # Write speculatively — overwritten if not stored
            result[count, j] = np.int32(val)

        ok = np.int64(total > 0) & np.int64(exact_mass >= orig_min) & np.int64(exact_mass <= orig_max)
        if do_rdbe_filter:
            ok &= np.int64(rdbe >= rdbe_min) & np.int64(rdbe <= rdbe_max)
            is_half_int = np.int64(2.0 * rdbe) & np.int64(1)
            ok &= charge_parity_int ^ is_half_int

        count += ok
    return result[:count]


# ========================================================================
# 3. RETURN PRECOMPUTED: return exact_mass/rdbe from Numba directly
# ========================================================================

@njit(fastmath=True)
def decomp_counts_only(c_arr, min_values, real_masses, rdbe_coeffs,
                       charge_mass_offset, orig_min, orig_max,
                       rdbe_min, rdbe_max, charge_parity_even,
                       do_rdbe_filter, max_results):
    """Return counts only — caller must recompute mass/rdbe in numpy."""
    n = len(real_masses)
    result = np.empty((max_results, n), dtype=np.int32)
    count = 0
    charge_parity_int = np.int64(charge_parity_even)
    for row in range(len(c_arr)):
        if count >= max_results:
            break
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]
            result[count, j] = np.int32(val)

        ok = np.int64(total > 0) & np.int64(exact_mass >= orig_min) & np.int64(exact_mass <= orig_max)
        if do_rdbe_filter:
            ok &= np.int64(rdbe >= rdbe_min) & np.int64(rdbe <= rdbe_max)
            is_half_int = np.int64(2.0 * rdbe) & np.int64(1)
            ok &= charge_parity_int ^ is_half_int
        count += ok
    counts = result[:count]

    # Simulate what caller must do: recompute mass & rdbe from counts
    counts_f = counts.astype(np.float64)
    exact_masses = np.empty(count, dtype=np.float64)
    rdbe_vals = np.empty(count, dtype=np.float64)
    for i in range(count):
        m = -charge_mass_offset
        r = 1.0
        for j in range(n):
            m += counts_f[i, j] * real_masses[j]
            r += counts_f[i, j] * rdbe_coeffs[j]
        exact_masses[i] = m
        rdbe_vals[i] = r
    return counts, exact_masses, rdbe_vals


@njit(fastmath=True)
def decomp_return_all(c_arr, min_values, real_masses, rdbe_coeffs,
                      charge_mass_offset, orig_min, orig_max,
                      rdbe_min, rdbe_max, charge_parity_even,
                      do_rdbe_filter, max_results):
    """Return counts + precomputed exact_mass + rdbe (no numpy recomp)."""
    n = len(real_masses)
    result = np.empty((max_results, n), dtype=np.int32)
    result_mass = np.empty(max_results, dtype=np.float64)
    result_rdbe = np.empty(max_results, dtype=np.float64)
    count = 0
    charge_parity_int = np.int64(charge_parity_even)
    for row in range(len(c_arr)):
        if count >= max_results:
            break
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            if do_rdbe_filter:
                rdbe += val_f * rdbe_coeffs[j]
            result[count, j] = np.int32(val)

        ok = np.int64(total > 0) & np.int64(exact_mass >= orig_min) & np.int64(exact_mass <= orig_max)
        if do_rdbe_filter:
            ok &= np.int64(rdbe >= rdbe_min) & np.int64(rdbe <= rdbe_max)
            is_half_int = np.int64(2.0 * rdbe) & np.int64(1)
            ok &= charge_parity_int ^ is_half_int

        result_mass[count] = exact_mass
        result_rdbe[count] = rdbe
        count += ok
    return result[:count], result_mass[:count], result_rdbe[:count]


# ========================================================================
# 4. FULL DECOMPOSITION: test with real FormulaFinder data
# ========================================================================

def bench_full_pipeline():
    """Compare current pipeline vs returning precomputed values."""
    from find_mfs import FormulaFinder
    from find_mfs.core.decomposer import MassDecomposer

    finder = FormulaFinder('CHNOPS')
    # Warm up
    finder.find_formulae(mass=100.0, error_ppm=5.0)

    configs = [
        ('500 Da / 5ppm / RDBE+octet', dict(mass=500.0, error_ppm=5.0, filter_rdbe=(0,20), check_octet=True)),
        ('750 Da / 5ppm / RDBE+octet', dict(mass=750.0, error_ppm=5.0, filter_rdbe=(0,25), check_octet=True, max_results=50000)),
    ]

    for label, kw in configs:
        times = []
        for i in range(15):
            t0 = time.perf_counter()
            r = finder.find_formulae(**kw)
            t1 = time.perf_counter()
            if i >= 3:
                times.append(t1 - t0)
        med = statistics.median(times) * 1000
        print(f"    {label:50s}: {med:7.2f} ms  ({len(r)} results)")


# ========================================================================
# RUNNER
# ========================================================================

def bench(name, fn, args, n_runs=100, warmup=10):
    for _ in range(warmup):
        result = fn(*args)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = statistics.median(times) * 1e6
    mn = min(times) * 1e6
    print(f"    {name:45s}: {med:8.1f} us  (min: {mn:.1f})")
    return med


def main():
    np.random.seed(42)

    n = 6
    n_cand = 5000
    # Realistic count arrays
    c_arr = np.random.randint(0, 20, size=(n_cand, n), dtype=np.int64)
    min_values = np.zeros(n, dtype=np.int64)
    real_m = np.array([1.00783, 12.0, 14.003, 15.995, 30.974, 31.972], dtype=np.float64)
    rdbe_c = np.array([-0.5, 1.0, 0.5, 0.0, 1.5, 0.0], dtype=np.float64)
    cmo = 0.0
    orig_min, orig_max = 400.0, 600.0

    leaf_args = (c_arr, min_values, real_m, rdbe_c, cmo, orig_min, orig_max,
                 0.0, 30.0, True, True, True)

    print("=" * 72)
    print("  BENCH 1: Branchy vs branchless leaf filter (5000 candidates)")
    print("=" * 72)
    t1 = bench("branchy (current)", leaf_branchy, leaf_args)
    t2 = bench("branchless (XOR + arith)", leaf_branchless, leaf_args)
    print(f"    Speedup: {t1/t2:.2f}x\n")

    write_args = (c_arr, min_values, real_m, rdbe_c, cmo, orig_min, orig_max,
                  0.0, 30.0, True, True, 10000)

    print("=" * 72)
    print("  BENCH 2: Separate write vs speculative write (5000 candidates)")
    print("=" * 72)
    t1 = bench("separate write loop", decomp_separate_write, write_args)
    t2 = bench("speculative write", decomp_speculative_write, write_args)
    print(f"    Speedup: {t1/t2:.2f}x\n")

    # Verify correctness
    r1 = decomp_separate_write(*write_args)
    r2 = decomp_speculative_write(*write_args)
    assert np.array_equal(r1, r2), f"Results differ! {r1.shape} vs {r2.shape}"
    print("    Correctness: PASS\n")

    print("=" * 72)
    print("  BENCH 3: Counts-only + numpy recomp vs return-all (5000 cand)")
    print("=" * 72)
    t1 = bench("counts-only + recompute", decomp_counts_only, write_args)
    t2 = bench("return precomputed mass/rdbe", decomp_return_all, write_args)
    print(f"    Speedup: {t1/t2:.2f}x\n")

    # Verify correctness
    c1, m1, r1 = decomp_counts_only(*write_args)
    c2, m2, r2 = decomp_return_all(*write_args)
    assert np.array_equal(c1, c2), "Counts differ!"
    assert np.allclose(m1, m2, atol=1e-10), f"Masses differ! max delta={np.max(np.abs(m1-m2))}"
    assert np.allclose(r1, r2, atol=1e-10), f"RDBE differ! max delta={np.max(np.abs(r1-r2))}"
    print("    Correctness: PASS\n")

    print("=" * 72)
    print("  BENCH 4: Speculative + branchless combined (5000 candidates)")
    print("=" * 72)
    t1 = bench("separate write + branchy", decomp_separate_write, write_args)
    t2 = bench("spec write + branchless + return-all", decomp_return_all, write_args)
    print(f"    Speedup: {t1/t2:.2f}x\n")

    print("=" * 72)
    print("  BENCH 5: Full find_formulae pipeline (end-to-end)")
    print("=" * 72)
    bench_full_pipeline()
    print()


if __name__ == "__main__":
    main()
