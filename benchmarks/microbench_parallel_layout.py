"""
Microbenchmarks for low-level optimization techniques:
1) Parallel loop lowering (Numba prange)
2) Memory layout and contiguity (C-order vs F-order)
3) Hot helper inlining impact on decomposition-style checks

Run:
    .venv/bin/python benchmarks/microbench_parallel_layout.py
"""
import os
import statistics
import time

# Keep runs deterministic and avoid persisted cache interactions.
os.environ['NUMBA_DISABLE_JIT_CACHE'] = '1'

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads


@njit(cache=False, inline='never')
def _is_decomposable_call(
    ERT: np.ndarray,
    i: int,
    m: int,
    a1: int,
) -> bool:
    if m < 0:
        return False
    return ERT[int(m % a1), i] <= m


@njit(cache=False, inline='always')
def _is_decomposable_inline(
    ERT: np.ndarray,
    i: int,
    m: int,
    a1: int,
) -> bool:
    if m < 0:
        return False
    return ERT[int(m % a1), i] <= m


@njit(cache=False, fastmath=True)
def decompose_full_call(
    ERT: np.ndarray,
    integer_masses: np.ndarray,
    real_masses: np.ndarray,
    bounds: np.ndarray,
    min_values: np.ndarray,
    min_int: int,
    max_int: int,
    original_min_mass: float,
    original_max_mass: float,
    charge_mass_offset: float,
    max_results: int,
    rdbe_coeffs: np.ndarray,
    rdbe_min: float,
    rdbe_max: float,
    check_octet: bool,
    charge_parity_even: bool,
    do_rdbe_filter: bool,
) -> int:
    """Decomposition kernel using a non-inlined decomposability helper."""
    a1 = integer_masses[0]
    num_elements = len(integer_masses)
    k = num_elements - 1
    kept = 0
    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and kept < max_results:
            if not _is_decomposable_call(ERT, i, m, a1):
                while i <= k and not _is_decomposable_call(ERT, i, m, a1):
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
                while i > 0 and _is_decomposable_call(ERT, i - 1, m, a1):
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
                            if do_rdbe_filter:
                                if rdbe < rdbe_min or rdbe > rdbe_max:
                                    pass
                                elif check_octet:
                                    is_half_int = (np.int64(2.0 * rdbe) & 1) == 1
                                    if charge_parity_even and not is_half_int:
                                        kept += 1
                                    elif (not charge_parity_even) and is_half_int:
                                        kept += 1
                                else:
                                    kept += 1
                            else:
                                kept += 1
                    i += 1
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return kept


@njit(cache=False, fastmath=True)
def decompose_full_inline(
    ERT: np.ndarray,
    integer_masses: np.ndarray,
    real_masses: np.ndarray,
    bounds: np.ndarray,
    min_values: np.ndarray,
    min_int: int,
    max_int: int,
    original_min_mass: float,
    original_max_mass: float,
    charge_mass_offset: float,
    max_results: int,
    rdbe_coeffs: np.ndarray,
    rdbe_min: float,
    rdbe_max: float,
    check_octet: bool,
    charge_parity_even: bool,
    do_rdbe_filter: bool,
) -> int:
    """Decomposition kernel using an always-inlined decomposability helper."""
    a1 = integer_masses[0]
    num_elements = len(integer_masses)
    k = num_elements - 1
    kept = 0
    c = np.zeros(num_elements, dtype=np.int64)

    for m_target in range(min_int, max_int + 1):
        for j in range(num_elements):
            c[j] = 0
        i = k
        m = np.int64(m_target)

        while i <= k and kept < max_results:
            if not _is_decomposable_inline(ERT, i, m, a1):
                while i <= k and not _is_decomposable_inline(ERT, i, m, a1):
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
                while i > 0 and _is_decomposable_inline(ERT, i - 1, m, a1):
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
                            if do_rdbe_filter:
                                if rdbe < rdbe_min or rdbe > rdbe_max:
                                    pass
                                elif check_octet:
                                    is_half_int = (np.int64(2.0 * rdbe) & 1) == 1
                                    if charge_parity_even and not is_half_int:
                                        kept += 1
                                    elif (not charge_parity_even) and is_half_int:
                                        kept += 1
                                else:
                                    kept += 1
                            else:
                                kept += 1
                    i += 1
                while i <= k and c[i] >= bounds[i]:
                    m += c[i] * integer_masses[i]
                    c[i] = 0
                    i += 1
                if i <= k:
                    m -= integer_masses[i]
                    c[i] += 1

    return kept


@njit(cache=False, fastmath=True)
def leaf_filter_serial_aos(
    counts: np.ndarray,
    min_values: np.ndarray,
    real_masses: np.ndarray,
    rdbe_coeffs: np.ndarray,
    charge_mass_offset: float,
    min_mass: float,
    max_mass: float,
    rdbe_min: float,
    rdbe_max: float,
    check_octet: bool,
    charge_parity_even: bool,
) -> int:
    """Row-wise serial kernel (array-of-structures style)."""
    n_rows, n_elem = counts.shape
    kept = 0
    for row in range(n_rows):
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n_elem):
            val = counts[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            rdbe += val_f * rdbe_coeffs[j]

        if total <= 0 or exact_mass < min_mass or exact_mass > max_mass:
            continue
        if rdbe < rdbe_min or rdbe > rdbe_max:
            continue
        if check_octet:
            is_half_int = (np.int64(2.0 * rdbe) & 1) == 1
            if charge_parity_even and is_half_int:
                continue
            if not charge_parity_even and not is_half_int:
                continue
        kept += 1
    return kept


@njit(cache=False, parallel=True, fastmath=True)
def leaf_filter_parallel_aos(
    counts: np.ndarray,
    min_values: np.ndarray,
    real_masses: np.ndarray,
    rdbe_coeffs: np.ndarray,
    charge_mass_offset: float,
    min_mass: float,
    max_mass: float,
    rdbe_min: float,
    rdbe_max: float,
    check_octet: bool,
    charge_parity_even: bool,
) -> int:
    """Row-wise parallel kernel lowered to parfor."""
    n_rows, n_elem = counts.shape
    keep = np.zeros(n_rows, dtype=np.uint8)

    for row in prange(n_rows):
        total = np.int64(0)
        exact_mass = -charge_mass_offset
        rdbe = 1.0
        for j in range(n_elem):
            val = counts[row, j] + min_values[j]
            val_f = np.float64(val)
            total += val
            exact_mass += val_f * real_masses[j]
            rdbe += val_f * rdbe_coeffs[j]

        ok = (
            total > 0
            and exact_mass >= min_mass
            and exact_mass <= max_mass
            and rdbe >= rdbe_min
            and rdbe <= rdbe_max
        )
        if ok and check_octet:
            is_half_int = (np.int64(2.0 * rdbe) & 1) == 1
            if charge_parity_even:
                ok = not is_half_int
            else:
                ok = is_half_int

        keep[row] = 1 if ok else 0

    kept = 0
    for row in range(n_rows):
        kept += keep[row]
    return kept


def bench(
    name: str,
    fn,
    args: tuple,
    n_runs: int = 20,
    warmup: int = 4,
    unit: str = 'us',
    inner_loops: int = 1,
) -> float:
    for _ in range(warmup):
        for _ in range(inner_loops):
            fn(*args)

    times = []
    result = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for _ in range(inner_loops):
            result = fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) / inner_loops)

    scale = 1e6 if unit == 'us' else 1e3
    med = statistics.median(times) * scale
    mn = min(times) * scale
    print(f"    {name:45s}: {med:9.3f} {unit}  (min: {mn:.3f})  result={result}")
    return med


def inspect_kernel_instructions(name: str, fn) -> None:
    """Print compact LLVM/ASM instruction summaries for one compiled signature."""
    if not fn.signatures:
        print(f"    {name}: no compiled signatures")
        return

    sig = fn.signatures[0]
    llvm = fn.inspect_llvm(sig)
    asm = fn.inspect_asm(sig)

    llvm_lines = llvm.splitlines()
    asm_lines = asm.splitlines()

    llvm_counts = {
        'call': sum(' call ' in ln for ln in llvm_lines),
        'sdiv/udiv': sum((' sdiv ' in ln) or (' udiv ' in ln) for ln in llvm_lines),
        'srem/urem': sum((' srem ' in ln) or (' urem ' in ln) for ln in llvm_lines),
        'fmul': sum(' fmul ' in ln for ln in llvm_lines),
        'fadd': sum(' fadd ' in ln for ln in llvm_lines),
        'br': sum(ln.strip().startswith('br ') for ln in llvm_lines),
        'parallel_for_refs': sum('parallel' in ln.lower() for ln in llvm_lines),
    }
    asm_counts = {
        'idiv': sum(ln.strip().startswith('idiv') for ln in asm_lines),
        'vfmadd': sum(ln.strip().startswith('vfmadd') for ln in asm_lines),
        'vmul*': sum(ln.strip().startswith(('vmulsd', 'vmulss')) for ln in asm_lines),
        'vadd*': sum(ln.strip().startswith(('vaddsd', 'vaddss')) for ln in asm_lines),
        'jmp/jcc': sum(ln.strip().startswith('j') for ln in asm_lines),
        'callq/call': sum(ln.strip().startswith(('callq', 'call')) for ln in asm_lines),
    }

    print(f"    {name}")
    print(f"      LLVM: {llvm_counts}")
    print(f"      ASM : {asm_counts}")


def main() -> None:
    np.random.seed(0)
    available_threads = get_num_threads()
    benchmark_threads = min(8, available_threads)
    set_num_threads(benchmark_threads)
    print(f"Using {benchmark_threads} Numba threads (available: {available_threads})")
    print()

    print("=" * 80)
    print("  BENCH 1: Hot Helper Inlining (Decomposition-Style Check Loop)")
    print("=" * 80)
    from find_mfs.core.decomposer import MassDecomposer
    from find_mfs.utils.filtering import BOND_ELECTRONS

    decomposer = MassDecomposer('CHNOPS')
    n_elem = len(decomposer.integer_masses)
    rdbe_coeffs_kernel = np.array(
        [0.5 * (BOND_ELECTRONS[s] - 2) for s in decomposer.element_symbols],
        dtype=np.float64,
    )
    mass = 750.0
    ppm = 5.0
    err = mass * ppm / 1e6
    orig_min = mass - err
    orig_max = mass + err
    bounds = np.full(n_elem, np.inf, dtype=np.float64)
    for i, element in enumerate(decomposer.elements):
        bounds[i] = min(bounds[i], float(np.floor(orig_max / element.mass)))
    min_values_decomp = np.zeros(n_elem, dtype=np.int64)
    min_int, max_int = decomposer._get_mass_range_as_integers(orig_min, orig_max)
    decomp_args = (
        decomposer.ERT,
        decomposer.integer_masses,
        decomposer.real_masses,
        bounds,
        min_values_decomp,
        min_int,
        max_int,
        orig_min,
        orig_max,
        0.0,
        50000,
        rdbe_coeffs_kernel,
        0.0,
        25.0,
        True,
        True,
        True,
    )

    t_call = bench(
        "full decompose (helper inline=never)",
        decompose_full_call,
        decomp_args,
        n_runs=20,
        warmup=4,
        unit='ms',
    )
    t_inline = bench(
        "full decompose (helper inline=always)",
        decompose_full_inline,
        decomp_args,
        n_runs=20,
        warmup=4,
        unit='ms',
    )
    print(f"    Speedup inline vs call: {t_call / t_inline:.3f}x")
    inspect_kernel_instructions("decompose_full_call", decompose_full_call)
    inspect_kernel_instructions("decompose_full_inline", decompose_full_inline)
    print()

    print("=" * 80)
    print("  BENCH 2: Memory Layout & Contiguity (Serial Row-Wise Leaf Kernel)")
    print("=" * 80)
    n_rows_layout = 2_000_000
    counts_c = np.random.randint(0, 20, size=(n_rows_layout, n_elem), dtype=np.int32)
    counts_f = np.asfortranarray(counts_c)
    min_values = np.zeros(n_elem, dtype=np.int64)
    real_masses = np.array([1.00783, 12.0, 14.003, 15.995, 30.974, 31.972], dtype=np.float64)
    rdbe_coeffs = np.array([-0.5, 1.0, 0.5, 0.0, 1.5, 0.0], dtype=np.float64)
    filter_args = (
        min_values,
        real_masses,
        rdbe_coeffs,
        0.0,   # charge_mass_offset
        400.0, # min_mass
        600.0, # max_mass
        0.0,   # rdbe_min
        30.0,  # rdbe_max
        True,  # check_octet
        True,  # charge_parity_even
    )

    t_c = bench(
        "C-order contiguous counts",
        leaf_filter_serial_aos,
        (counts_c, *filter_args),
        n_runs=8,
        warmup=3,
        unit='ms',
    )
    t_f = bench(
        "F-order non-row-contiguous counts",
        leaf_filter_serial_aos,
        (counts_f, *filter_args),
        n_runs=8,
        warmup=3,
        unit='ms',
    )
    print(f"    Speedup C-order vs F-order: {t_f / t_c:.3f}x")
    inspect_kernel_instructions("leaf_filter_serial_aos", leaf_filter_serial_aos)
    print()

    print("=" * 80)
    print("  BENCH 3: Parallel Loop Lowering Threshold (prange)")
    print("=" * 80)
    for n_rows in (200_000, 3_000_000):
        counts = np.random.randint(0, 25, size=(n_rows, n_elem), dtype=np.int32)
        args = (counts, *filter_args)
        print(f"  Dataset size: {n_rows:,} rows")
        t_serial = bench(
            "serial leaf filter",
            leaf_filter_serial_aos,
            args,
            n_runs=10 if n_rows < 1_000_000 else 5,
            warmup=2,
            unit='ms',
            inner_loops=25 if n_rows < 1_000_000 else 3,
        )
        t_parallel = bench(
            "parallel leaf filter (prange)",
            leaf_filter_parallel_aos,
            args,
            n_runs=10 if n_rows < 1_000_000 else 5,
            warmup=2,
            unit='ms',
            inner_loops=25 if n_rows < 1_000_000 else 3,
        )
        print(f"    Speedup parallel vs serial: {t_serial / t_parallel:.3f}x")
        print()

    inspect_kernel_instructions("leaf_filter_parallel_aos", leaf_filter_parallel_aos)
    print()

    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(
        "  - Inlining hot helpers in decomposition loops consistently improves speed.\n"
        "  - C-order row contiguity is important for row-wise candidate kernels.\n"
        "  - prange parallel lowering can substantially speed numeric filtering,\n"
        "    but should be benchmarked on target hardware/thread settings."
    )


if __name__ == "__main__":
    main()
