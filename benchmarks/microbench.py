"""
Microbenchmarks for potential optimizations identified from LLVM/ASM analysis.

Candidates:
1. Float remainder `(2*rdbe) % 2.0 == 1.0` compiles to a Python-level
   _ZN05numba6python3remE6double function call. Replace with integer check.
2. `m % a1` in _is_decomposable uses expensive idiv (~20-90 cycles).
   Use magic-number multiply trick for fixed a1.
3. Manual unroll of 6-element dot product vs loop.
4. Precompute `c[j] + min_values[j]` once and reuse for both exact_mass
   and RDBE computation.
"""
import os
os.environ['NUMBA_DISABLE_JIT_CACHE'] = '1'

import time
import statistics
import numpy as np
from numba import njit


# ========================================================================
# 1. OCTET CHECK: float remainder vs integer cast
# ========================================================================

@njit
def octet_float_rem(rdbe_values, charge_parity_even):
    """Current approach: (2.0 * rdbe) % 2.0 == 1.0"""
    count = 0
    for i in range(len(rdbe_values)):
        doubled = 2.0 * rdbe_values[i]
        is_half_int = (doubled % 2.0) == 1.0
        if charge_parity_even and is_half_int:
            count += 1
        elif not charge_parity_even and not is_half_int:
            count += 1
    return count


@njit
def octet_int_cast(rdbe_values, charge_parity_even):
    """Proposed: cast 2*rdbe to int64, check & 1"""
    count = 0
    for i in range(len(rdbe_values)):
        doubled_int = np.int64(2.0 * rdbe_values[i])
        is_half_int = (doubled_int & 1) == 1
        if charge_parity_even and is_half_int:
            count += 1
        elif not charge_parity_even and not is_half_int:
            count += 1
    return count


@njit
def octet_frac_check(rdbe_values, charge_parity_even):
    """Alternative: check fractional part directly via floor"""
    count = 0
    for i in range(len(rdbe_values)):
        rdbe = rdbe_values[i]
        frac = rdbe - np.floor(rdbe)
        # is_half_int when frac is ~0.5 (not 0.0)
        is_half_int = frac > 0.25
        if charge_parity_even and is_half_int:
            count += 1
        elif not charge_parity_even and not is_half_int:
            count += 1
    return count


# ========================================================================
# 2. MODULO: idiv vs magic multiply for _is_decomposable
# ========================================================================

@njit
def modulo_idiv(ERT, masses, m_values, a1):
    """Current: m % a1 using idiv instruction"""
    count = 0
    k = len(masses) - 1
    for m in m_values:
        r = m % a1
        if ERT[r, k] <= m:
            count += 1
    return count


@njit
def modulo_precomp(ERT, masses, m_values, a1):
    """Proposed: precompute via subtract-and-compare loop.
    Only viable if a1 is small relative to m."""
    count = 0
    k = len(masses) - 1
    for m in m_values:
        # Use builtin % — but let's see if restructuring helps
        r = np.int64(m - (m // a1) * a1)
        if ERT[r, k] <= m:
            count += 1
    return count


# ========================================================================
# 3. DOT PRODUCT: loop vs manual unroll for N=6 (CHNOPS)
# ========================================================================

@njit(fastmath=True)
def dot_loop(c, min_values, real_masses, rdbe_coeffs, n_iter):
    """Current: loop over num_elements"""
    total_mass = 0.0
    total_rdbe = 0.0
    for _ in range(n_iter):
        exact_mass = 0.0
        rdbe = 1.0
        total = np.int64(0)
        for j in range(len(c)):
            val = c[j] + min_values[j]
            total += val
            exact_mass += val * real_masses[j]
            rdbe += val * rdbe_coeffs[j]
        total_mass += exact_mass
        total_rdbe += rdbe
    return total_mass, total_rdbe


@njit(fastmath=True)
def dot_combined(c, min_values, real_masses, rdbe_coeffs, n_iter):
    """Proposed: precompute vals array once, then two dot products"""
    n = len(c)
    vals = np.empty(n, dtype=np.float64)
    total_mass = 0.0
    total_rdbe = 0.0
    for _ in range(n_iter):
        total = np.int64(0)
        for j in range(n):
            v = c[j] + min_values[j]
            vals[j] = np.float64(v)
            total += v
        exact_mass = 0.0
        rdbe = 1.0
        for j in range(n):
            exact_mass += vals[j] * real_masses[j]
            rdbe += vals[j] * rdbe_coeffs[j]
        total_mass += exact_mass
        total_rdbe += rdbe
    return total_mass, total_rdbe


@njit(fastmath=True)
def dot_single_pass(c, min_values, real_masses, rdbe_coeffs, n_iter):
    """Alternative: single pass computing both products simultaneously"""
    total_mass = 0.0
    total_rdbe = 0.0
    for _ in range(n_iter):
        exact_mass = 0.0
        rdbe = 1.0
        total = np.int64(0)
        for j in range(len(c)):
            val_f = np.float64(c[j] + min_values[j])
            total += c[j] + min_values[j]
            exact_mass += val_f * real_masses[j]
            rdbe += val_f * rdbe_coeffs[j]
        total_mass += exact_mass
        total_rdbe += rdbe
    return total_mass, total_rdbe


# ========================================================================
# 4. FULL LEAF: simulate the complete leaf computation path
# ========================================================================

@njit(fastmath=True)
def leaf_current(c_arr, min_values, real_masses, rdbe_coeffs,
                 charge_mass_offset, rdbe_min, rdbe_max,
                 check_octet, charge_parity_even):
    """Current leaf implementation (exact_mass loop + separate RDBE loop
    with float remainder)"""
    count = 0
    num_elements = len(real_masses)
    for row in range(len(c_arr)):
        # Exact mass computation
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        for j in range(num_elements):
            val = c_arr[row, j] + min_values[j]
            total += val
            exact_mass += val * real_masses[j]

        if total <= 0:
            continue

        # RDBE computation + filter
        rdbe = 1.0
        for j in range(num_elements):
            rdbe += (c_arr[row, j] + min_values[j]) * rdbe_coeffs[j]
        if rdbe < rdbe_min or rdbe > rdbe_max:
            continue

        # Octet check (float remainder)
        if check_octet:
            doubled = 2.0 * rdbe
            is_half_int = (doubled % 2.0) == 1.0
            if charge_parity_even and is_half_int:
                continue
            if not charge_parity_even and not is_half_int:
                continue

        count += 1
    return count


@njit(fastmath=True)
def leaf_optimized(c_arr, min_values, real_masses, rdbe_coeffs,
                   charge_mass_offset, rdbe_min, rdbe_max,
                   check_octet, charge_parity_even):
    """Optimized: single pass for mass+RDBE, integer octet check"""
    count = 0
    num_elements = len(real_masses)
    for row in range(len(c_arr)):
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(num_elements):
            val = c_arr[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            rdbe += val_f * rdbe_coeffs[j]

        if total <= 0:
            continue

        if rdbe < rdbe_min or rdbe > rdbe_max:
            continue

        if check_octet:
            doubled_int = np.int64(2.0 * rdbe)
            is_half_int = (doubled_int & 1) == 1
            if charge_parity_even and is_half_int:
                continue
            if not charge_parity_even and not is_half_int:
                continue

        count += 1
    return count


# ========================================================================
# Runner
# ========================================================================

def bench(name, fn, args, n_runs=50, warmup=5):
    """Run a microbenchmark."""
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        result = fn(*args)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = statistics.median(times) * 1e6  # microseconds
    mn = min(times) * 1e6
    mx = max(times) * 1e6
    print(f"    {name:40s}: {med:8.1f} us  (min: {mn:.1f}, max: {mx:.1f})  result={result}")
    return med


def main():
    np.random.seed(42)

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  BENCHMARK 1: Octet check — float remainder vs integer")
    print("=" * 72)
    # Generate realistic RDBE values (multiples of 0.5 for CHNOPS)
    rdbe_values = np.array([float(x) / 2.0 for x in range(0, 2000)], dtype=np.float64)

    t1 = bench("float_rem (current)", octet_float_rem, (rdbe_values, True))
    t2 = bench("int_cast & 1", octet_int_cast, (rdbe_values, True))
    t3 = bench("floor frac check", octet_frac_check, (rdbe_values, True))
    print(f"    Speedup int_cast vs float_rem: {t1/t2:.2f}x")
    print(f"    Speedup frac vs float_rem: {t1/t3:.2f}x")
    print()

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  BENCHMARK 2: Modulo — idiv vs alternatives")
    print("=" * 72)
    n_elem = 6
    # Simulate ERT table for a1 = 1008 (typical for CHNOPS H)
    a1 = np.int64(1008)
    ERT = np.full((a1, n_elem), 500.0, dtype=np.float64)
    ERT[0, :] = 0
    masses = np.arange(n_elem, dtype=np.int64)
    m_values = np.arange(5000, 15000, dtype=np.int64)

    t1 = bench("idiv (current)", modulo_idiv, (ERT, masses, m_values, a1))
    t2 = bench("div+mul (explicit)", modulo_precomp, (ERT, masses, m_values, a1))
    print(f"    Speedup: {t1/t2:.2f}x")
    print()

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  BENCHMARK 3: Dot product strategies (N=6, CHNOPS)")
    print("=" * 72)
    c = np.array([6, 12, 0, 6, 0, 0], dtype=np.int64)  # glucose-like
    min_v = np.zeros(6, dtype=np.int64)
    real_m = np.array([1.00783, 12.0, 14.003, 15.995, 30.974, 31.972], dtype=np.float64)
    rdbe_c = np.array([-0.5, 1.0, 0.5, 0.0, 1.5, 0.0], dtype=np.float64)
    n_iter = 10000

    t1 = bench("loop (current)", dot_loop, (c, min_v, real_m, rdbe_c, n_iter))
    t2 = bench("precompute vals", dot_combined, (c, min_v, real_m, rdbe_c, n_iter))
    t3 = bench("single pass fused", dot_single_pass, (c, min_v, real_m, rdbe_c, n_iter))
    print(f"    Speedup precompute vs loop: {t1/t2:.2f}x")
    print(f"    Speedup single_pass vs loop: {t1/t3:.2f}x")
    print()

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  BENCHMARK 4: Full leaf path — current vs optimized")
    print("=" * 72)
    # Generate realistic count arrays (2000 candidate rows, 6 elements)
    n_candidates = 2000
    c_arr = np.random.randint(0, 15, size=(n_candidates, 6), dtype=np.int64)
    min_values = np.zeros(6, dtype=np.int64)
    charge_mass_offset = 0.0

    t1 = bench("leaf_current", leaf_current,
               (c_arr, min_values, real_m, rdbe_c, charge_mass_offset,
                0.0, 30.0, True, True))
    t2 = bench("leaf_optimized", leaf_optimized,
               (c_arr, min_values, real_m, rdbe_c, charge_mass_offset,
                0.0, 30.0, True, True))
    print(f"    Speedup optimized vs current: {t1/t2:.2f}x")
    print()

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  BENCHMARK 5: Full leaf — WITHOUT octet (RDBE-only filter)")
    print("=" * 72)
    t1 = bench("leaf_current (no octet)", leaf_current,
               (c_arr, min_values, real_m, rdbe_c, charge_mass_offset,
                0.0, 30.0, False, True))
    t2 = bench("leaf_optimized (no octet)", leaf_optimized,
               (c_arr, min_values, real_m, rdbe_c, charge_mass_offset,
                0.0, 30.0, False, True))
    print(f"    Speedup optimized vs current: {t1/t2:.2f}x")
    print()

    # -------------------------------------------------------------------
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print("""
    Key findings:
    - Benchmark 1 tests the octet half-integer check in isolation
    - Benchmark 2 tests modulo alternatives for _is_decomposable
    - Benchmark 3 tests dot product strategies for the exact mass + RDBE inner loop
    - Benchmark 4 tests the complete leaf path with all optimizations combined
    - Benchmark 5 tests the leaf path with RDBE filtering only (no octet)
    """)


if __name__ == "__main__":
    main()
