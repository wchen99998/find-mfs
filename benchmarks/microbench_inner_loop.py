"""
Microbenchmarks for inner-loop optimizations in the full decomposition.

Key insight from previous benchmarks:
- Branchless leaf filter: SLOWER (0.83x) — CPU predicts well
- Speculative writes: SLOWER (0.39x) — wasted memory traffic
- Return precomputed mass/rdbe: FASTER (1.15x) — saves numpy recomp

New ideas to test:
1. Remove `if do_rdbe_filter` from inner loop by always computing RDBE
   (use -inf/+inf bounds when not filtering — the check becomes free)
2. Return precomputed mass/rdbe from the FULL decomposition function
3. Branchless `count +=` via arithmetic instead of `if store: count += 1`
4. Precompute vals array to avoid recomputing c[j]+min_values[j] for write
"""
import os
os.environ['NUMBA_DISABLE_JIT_CACHE'] = '1'

import time
import statistics
import numpy as np
from numba import njit


# ========================================================================
# Full decomposition variants using realistic ERT data
# ========================================================================

@njit(cache=False)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=False, fastmath=True)
def decomp_current(ERT, integer_masses, real_masses, bounds, min_values,
                   min_int, max_int, original_min_mass, original_max_mass,
                   charge_mass_offset, max_results,
                   rdbe_coeffs, rdbe_min, rdbe_max,
                   check_octet, charge_parity_even, do_rdbe_filter):
    """Current implementation: conditional RDBE in inner loop, return counts only."""
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1
    result = np.empty((max_results, num_elements), dtype=np.int32)
    count = 0
    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and count < max_results:
            if m < 0 or ERT[int(m % a1), i] > m:
                while i <= k and (m < 0 or ERT[int(m % a1), i] > m):
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
                while i > 0 and m >= 0 and ERT[int(m % a1), i - 1] <= m:
                    i -= 1
                if i == 0:
                    c[0] = m // a1
                    if c[0] <= bounds[0]:
                        total = np.int64(0)
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        for j in range(num_elements):
                            val = c[j] + min_values[j]
                            val_f = np.float64(val)
                            total += val
                            exact_mass += val_f * real_masses[j]
                            if do_rdbe_filter:
                                rdbe += val_f * rdbe_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
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


@njit(cache=False, fastmath=True)
def decomp_always_rdbe(ERT, integer_masses, real_masses, bounds, min_values,
                       min_int, max_int, original_min_mass, original_max_mass,
                       charge_mass_offset, max_results,
                       rdbe_coeffs, rdbe_min, rdbe_max,
                       check_octet, charge_parity_even, do_rdbe_filter):
    """Always compute RDBE (remove if-branch from inner loop).
    When not filtering, coeffs=0 and bounds=-inf/+inf make it a no-op."""
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1
    result = np.empty((max_results, num_elements), dtype=np.int32)
    count = 0
    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and count < max_results:
            if m < 0 or ERT[int(m % a1), i] > m:
                while i <= k and (m < 0 or ERT[int(m % a1), i] > m):
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
                while i > 0 and m >= 0 and ERT[int(m % a1), i - 1] <= m:
                    i -= 1
                if i == 0:
                    c[0] = m // a1
                    if c[0] <= bounds[0]:
                        total = np.int64(0)
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        for j in range(num_elements):
                            val = c[j] + min_values[j]
                            val_f = np.float64(val)
                            total += val
                            exact_mass += val_f * real_masses[j]
                            # Always compute — branch removed from inner loop
                            rdbe += val_f * rdbe_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
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


@njit(cache=False, fastmath=True)
def decomp_return_precomputed(ERT, integer_masses, real_masses, bounds, min_values,
                              min_int, max_int, original_min_mass, original_max_mass,
                              charge_mass_offset, max_results,
                              rdbe_coeffs, rdbe_min, rdbe_max,
                              check_octet, charge_parity_even, do_rdbe_filter):
    """Return counts + precomputed exact_mass and rdbe arrays.
    Always compute RDBE (branch-free inner loop)."""
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1
    result = np.empty((max_results, num_elements), dtype=np.int32)
    result_mass = np.empty(max_results, dtype=np.float64)
    result_rdbe = np.empty(max_results, dtype=np.float64)
    count = 0
    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and count < max_results:
            if m < 0 or ERT[int(m % a1), i] > m:
                while i <= k and (m < 0 or ERT[int(m % a1), i] > m):
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
                while i > 0 and m >= 0 and ERT[int(m % a1), i - 1] <= m:
                    i -= 1
                if i == 0:
                    c[0] = m // a1
                    if c[0] <= bounds[0]:
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
                                result_mass[count] = exact_mass
                                result_rdbe[count] = rdbe
                                count += 1
                    i += 1
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return result[:count], result_mass[:count], result_rdbe[:count]


@njit(cache=False, fastmath=True)
def decomp_precompute_vals(ERT, integer_masses, real_masses, bounds, min_values,
                           min_int, max_int, original_min_mass, original_max_mass,
                           charge_mass_offset, max_results,
                           rdbe_coeffs, rdbe_min, rdbe_max,
                           check_octet, charge_parity_even, do_rdbe_filter):
    """Precompute vals array to avoid recomputing c[j]+min_values[j] for write.
    Also returns precomputed mass/rdbe."""
    num_elements = len(integer_masses)
    a1 = integer_masses[0]
    k = num_elements - 1
    result = np.empty((max_results, num_elements), dtype=np.int32)
    result_mass = np.empty(max_results, dtype=np.float64)
    result_rdbe = np.empty(max_results, dtype=np.float64)
    count = 0
    c = np.zeros(num_elements, dtype=np.int64)
    vals = np.empty(num_elements, dtype=np.int32)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and count < max_results:
            if m < 0 or ERT[int(m % a1), i] > m:
                while i <= k and (m < 0 or ERT[int(m % a1), i] > m):
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
                while i > 0 and m >= 0 and ERT[int(m % a1), i - 1] <= m:
                    i -= 1
                if i == 0:
                    c[0] = m // a1
                    if c[0] <= bounds[0]:
                        total = np.int64(0)
                        exact_mass = -charge_mass_offset
                        rdbe = 1.0
                        for j in range(num_elements):
                            val = np.int32(c[j] + min_values[j])
                            vals[j] = val
                            val_f = np.float64(val)
                            total += val
                            exact_mass += val_f * real_masses[j]
                            rdbe += val_f * rdbe_coeffs[j]

                        if total > 0 and original_min_mass <= exact_mass <= original_max_mass:
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
                                result[count] = vals
                                result_mass[count] = exact_mass
                                result_rdbe[count] = rdbe
                                count += 1
                    i += 1
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return result[:count], result_mass[:count], result_rdbe[:count]


# ========================================================================
# Runner
# ========================================================================

def bench(name, fn, args, n_runs=20, warmup=5):
    for _ in range(warmup):
        result = fn(*args)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med = statistics.median(times) * 1e3
    mn = min(times) * 1e3
    # get result count
    if isinstance(result, tuple):
        n_res = len(result[0])
    else:
        n_res = len(result)
    print(f"    {name:50s}: {med:7.3f} ms  (min: {mn:.3f})  n={n_res}")
    return med


def main():
    from find_mfs.core.decomposer import MassDecomposer

    decomposer = MassDecomposer('CHNOPS')

    # Extract real ERT data and element info
    from find_mfs.utils.filtering import BOND_ELECTRONS
    symbols = decomposer.element_symbols
    rdbe_coeffs = np.array(
        [0.5 * (BOND_ELECTRONS[s] - 2) for s in symbols], dtype=np.float64
    )

    # Get real arrays from decomposer
    ERT = decomposer.ERT
    integer_masses = decomposer.integer_masses
    real_masses = decomposer.real_masses

    configs = [
        {
            'label': '500 Da / 5ppm / NO filter',
            'mass': 500.0, 'ppm': 5.0,
            'rdbe_min': -np.inf, 'rdbe_max': np.inf,
            'check_octet': False, 'do_rdbe_filter': False,
        },
        {
            'label': '500 Da / 5ppm / RDBE(0,20)+octet',
            'mass': 500.0, 'ppm': 5.0,
            'rdbe_min': 0.0, 'rdbe_max': 20.0,
            'check_octet': True, 'do_rdbe_filter': True,
        },
        {
            'label': '750 Da / 5ppm / RDBE(0,25)+octet',
            'mass': 750.0, 'ppm': 5.0,
            'rdbe_min': 0.0, 'rdbe_max': 25.0,
            'check_octet': True, 'do_rdbe_filter': True,
        },
    ]

    import math
    from molmass.elements import ELECTRON

    for cfg in configs:
        mass = cfg['mass']
        ppm = cfg['ppm']
        charge = 0

        adjusted_mass = mass + ELECTRON.mass * charge

        # Compute mass ranges (mirroring decomposer logic)
        error = mass * ppm / 1e6
        orig_min = adjusted_mass - error
        orig_max = adjusted_mass + error
        min_mass = orig_min
        max_mass = orig_max

        # Tighten bounds
        bounds = [float('inf')] * len(symbols)
        min_values = [0] * len(symbols)
        for i_e, element in enumerate(decomposer.elements):
            mass_bound = math.floor(max_mass / element.mass)
            bounds[i_e] = min(bounds[i_e], float(mass_bound))

        from_int = math.ceil((1 + decomposer.min_error) * min_mass / decomposer.precision)
        to_int = math.floor((1 + decomposer.max_error) * max_mass / decomposer.precision)
        min_int = max(0, int(from_int))
        max_int = max(min_int, int(to_int))

        bounds_arr = np.array(bounds, dtype=np.float64)
        min_values_arr = np.array(min_values, dtype=np.int64)
        charge_mass_offset = ELECTRON.mass * charge
        max_results = 50000

        # For always-rdbe version when not filtering, use zero coeffs
        if cfg['do_rdbe_filter']:
            rdbe_c = rdbe_coeffs
        else:
            rdbe_c = rdbe_coeffs  # same coeffs, but do_rdbe_filter=False skips check

        # For always-rdbe: when not filtering, use zero coeffs + -inf/+inf bounds
        if not cfg['do_rdbe_filter']:
            rdbe_c_zero = np.zeros_like(rdbe_coeffs)
        else:
            rdbe_c_zero = rdbe_coeffs

        common = (ERT, integer_masses, real_masses, bounds_arr, min_values_arr,
                  min_int, max_int, orig_min, orig_max, charge_mass_offset,
                  max_results, rdbe_c,
                  cfg['rdbe_min'], cfg['rdbe_max'],
                  cfg['check_octet'], True, cfg['do_rdbe_filter'])

        common_always = (ERT, integer_masses, real_masses, bounds_arr, min_values_arr,
                         min_int, max_int, orig_min, orig_max, charge_mass_offset,
                         max_results, rdbe_c_zero,
                         cfg['rdbe_min'], cfg['rdbe_max'],
                         cfg['check_octet'], True, cfg['do_rdbe_filter'])

        print(f"\n{'='*72}")
        print(f"  {cfg['label']}")
        print(f"{'='*72}")
        t1 = bench("current (conditional RDBE in loop)", decomp_current, common)
        t2 = bench("always-RDBE (branch-free inner loop)", decomp_always_rdbe, common_always if not cfg['do_rdbe_filter'] else common)
        t3 = bench("return precomputed mass/rdbe", decomp_return_precomputed, common_always if not cfg['do_rdbe_filter'] else common)
        t4 = bench("precompute vals + return all", decomp_precompute_vals, common_always if not cfg['do_rdbe_filter'] else common)

        print(f"    {'':50s}  always-rdbe: {t1/t2:.3f}x")
        print(f"    {'':50s}  return-pre:  {t1/t3:.3f}x")
        print(f"    {'':50s}  precomp-val: {t1/t4:.3f}x")

        # Verify correctness
        r_ref = decomp_current(*common)
        r_new = decomp_always_rdbe(*(common_always if not cfg['do_rdbe_filter'] else common))
        r_pre, _, _ = decomp_return_precomputed(*(common_always if not cfg['do_rdbe_filter'] else common))
        r_pv, _, _ = decomp_precompute_vals(*(common_always if not cfg['do_rdbe_filter'] else common))
        assert np.array_equal(r_ref, r_new), f"always-rdbe mismatch: {r_ref.shape} vs {r_new.shape}"
        assert np.array_equal(r_ref, r_pre), f"return-pre mismatch: {r_ref.shape} vs {r_pre.shape}"
        assert np.array_equal(r_ref, r_pv), f"precomp-val mismatch: {r_ref.shape} vs {r_pv.shape}"
        print("    Correctness: ALL PASS")


if __name__ == "__main__":
    main()
