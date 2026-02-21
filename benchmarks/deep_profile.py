#!/usr/bin/env python3
"""
Deep profiling to identify low-hanging performance improvements.

Breaks down timing of every stage in the pipeline to find bottlenecks.
"""

import time
import numpy as np
from molmass import Formula

from find_mfs import FormulaFinder
from find_mfs.isotopes.config import SingleEnvelopeMatch
from find_mfs.isotopes.envelope import (
    get_isotope_envelope,
    match_isotope_envelope,
    match_isotope_envelope_fast,
    _get_scorer,
)
from find_mfs.isotopes._isospec_bridge import get_isotope_arrays
from find_mfs.core.algorithms import _decompose_mass_range


def time_fn(fn, n=200, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times)//2]  # median


def profile_old_path_breakdown():
    """Break down old match_isotope_envelope into its components."""
    print("=" * 70)
    print("OLD PATH BREAKDOWN (match_isotope_envelope)")
    print("=" * 70)

    formula_str = "C31H36N2O11"
    f = Formula(formula_str)
    envelope = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)

    # 1. IsoSpecPy IsoThreshold call
    import IsoSpecPy as iso
    def step_isospec():
        iso.IsoThreshold(formula=formula_str, threshold=0.001)
    t_isospec = time_fn(step_isospec)

    # 2. Charge stripping + formula parsing overhead
    def step_formula_parse():
        Formula(formula_str)
    t_parse = time_fn(step_formula_parse)

    # 3. combine_unresolved_isotopologues
    from find_mfs.isotopes.envelope import combine_unresolved_isotopologues, rescale_envelope
    raw_isologues = []
    calc = iso.IsoThreshold(formula=formula_str, threshold=0.001)
    for mass, prob in calc:
        raw_isologues.append((mass, prob))
    raw_arr = np.array(raw_isologues, dtype=np.float32)

    def step_combine():
        combine_unresolved_isotopologues(raw_arr, mz_tolerance=0.05)
    t_combine = time_fn(step_combine)

    # 4. Peak matching loop (Python)
    sim_env = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)
    obs_env = envelope.copy()
    def step_match():
        n = obs_env.shape[0]
        pred = np.zeros(n, dtype=np.float64)
        matches = np.full(n, False)
        for i in range(n):
            diffs = np.abs(sim_env[:, 0] - obs_env[i, 0])
            closest = np.argmin(diffs)
            if diffs[closest] <= 0.01:
                pred[i] = sim_env[closest, 1]
                matches[i] = True
    t_match = time_fn(step_match)

    # 5. Full old path
    t_full = time_fn(lambda: match_isotope_envelope(
        f, envelope, 0.01, 0.05, 0.001))

    print(f"  Formula parse:    {t_parse*1e6:7.0f} µs")
    print(f"  IsoSpecPy call:   {t_isospec*1e6:7.0f} µs")
    print(f"  Combine peaks:    {t_combine*1e6:7.0f} µs")
    print(f"  Peak matching:    {t_match*1e6:7.0f} µs")
    print(f"  Full old path:    {t_full*1e6:7.0f} µs")
    print(f"  Unaccounted:      {(t_full - t_isospec - t_combine - t_match)*1e6:7.0f} µs")


def profile_new_path_breakdown():
    """Break down new match_isotope_envelope_fast into its components."""
    print("\n" + "=" * 70)
    print("NEW PATH BREAKDOWN (match_isotope_envelope_fast)")
    print("=" * 70)

    formula_str = "C31H36N2O11"
    f = Formula(formula_str)
    envelope = get_isotope_envelope(f, mz_tolerance=0.05, threshold=0.001)

    comp = f.composition()
    symbols = []
    counts = []
    for element, item in sorted(comp.items()):
        symbols.append(element)
        counts.append(item.count)

    # Prepare arrays
    iso_numbers, flat_masses, flat_probs = get_isotope_arrays(symbols)
    atom_counts = np.array(counts, dtype=np.int32)
    obs_mz = np.ascontiguousarray(envelope[:, 0], dtype=np.float64)
    obs_int = np.ascontiguousarray(envelope[:, 1], dtype=np.float64)

    scorer = _get_scorer()
    from find_mfs.isotopes.envelope import ELECTRON

    # 1. get_isotope_arrays call
    def step_arrays():
        get_isotope_arrays(symbols)
    t_arrays = time_fn(step_arrays)

    # 2. Array preparation (ascontiguousarray, etc.)
    def step_prep():
        active_s = [s for s, c in zip(symbols, counts) if c > 0]
        active_c = [c for c in counts if c > 0]
        np.array(active_c, dtype=np.int32)
        np.ascontiguousarray(envelope[:, 0], dtype=np.float64)
        np.ascontiguousarray(envelope[:, 1], dtype=np.float64)
    t_prep = time_fn(step_prep)

    # 3. Pure Numba scorer call
    def step_scorer():
        scorer(
            iso_numbers, atom_counts, flat_masses, flat_probs, len(symbols),
            obs_mz, obs_int, 0.05, 0.01, 0.001, 0, ELECTRON.mass,
        )
    t_scorer = time_fn(step_scorer)

    # 4. Full new path
    t_full = time_fn(lambda: match_isotope_envelope_fast(
        symbols, counts, 0, envelope, 0.01, 0.05, 0.001))

    print(f"  get_isotope_arrays:  {t_arrays*1e6:7.0f} µs")
    print(f"  Array preparation:   {t_prep*1e6:7.0f} µs")
    print(f"  Numba scorer:        {t_scorer*1e6:7.0f} µs")
    print(f"  Full new path:       {t_full*1e6:7.0f} µs")
    print(f"  Unaccounted:         {(t_full - t_arrays - t_prep - t_scorer)*1e6:7.0f} µs")


def profile_pipeline_breakdown():
    """Break down the full find_formulae pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE BREAKDOWN (find_formulae, 612 Da)")
    print("=" * 70)

    finder = FormulaFinder("CHNOPS")
    mass = 612.211
    charge = 1

    envelope = np.array([
        [612.211, 1.0],
        [613.215, 0.34],
        [614.218, 0.06],
    ])

    # 1. Decomposition only (no filters)
    def step_decompose_only():
        finder.decomposer.decompose_to_counts(
            query_mass=mass, charge=charge,
            ppm_error=5.0, max_results=10000,
        )
    t_decompose = time_fn(step_decompose_only, n=100)

    # 2. Decomposition + RDBE/octet
    rdbe_coeffs = finder._rdbe_coeffs
    def step_decompose_rdbe():
        finder.decomposer.decompose_to_counts(
            query_mass=mass, charge=charge,
            ppm_error=5.0, max_results=10000,
            rdbe_coeffs=rdbe_coeffs,
            rdbe_min=-0.5, rdbe_max=40.0,
            check_octet=True,
        )
    t_decompose_rdbe = time_fn(step_decompose_rdbe, n=100)

    # 3. Decomposition + RDBE + isotope prefilter
    from find_mfs.isotopes._isospec_bridge import M1_RATIOS, M2_DIRECT
    m1_coeffs = finder._iso_m1_coeffs
    m2_coeffs = finder._iso_m2_direct_coeffs

    obs_env = envelope.copy()
    obs_env[:, 1] /= obs_env[:, 1].max()
    base_mz = obs_env[np.argmax(obs_env[:, 1]), 0]
    obs_m1 = 0.0
    obs_m2 = 0.0
    for i in range(obs_env.shape[0]):
        delta = obs_env[i, 0] - base_mz
        if 0.9 <= delta <= 1.1:
            obs_m1 = obs_env[i, 1]
        elif 1.9 <= delta <= 2.1:
            obs_m2 = obs_env[i, 1]

    def step_decompose_all():
        finder.decomposer.decompose_to_counts(
            query_mass=mass, charge=charge,
            ppm_error=5.0, max_results=10000,
            rdbe_coeffs=rdbe_coeffs,
            rdbe_min=-0.5, rdbe_max=40.0,
            check_octet=True,
            iso_m1_coeffs=m1_coeffs,
            iso_m2_direct_coeffs=m2_coeffs,
            obs_m1_ratio=obs_m1, obs_m2_ratio=obs_m2,
        )
    t_decompose_all = time_fn(step_decompose_all, n=100)

    # Count candidates at each stage
    counts_raw, _ = finder.decomposer.decompose_to_counts(
        query_mass=mass, charge=charge, ppm_error=5.0, max_results=10000)
    counts_rdbe, _ = finder.decomposer.decompose_to_counts(
        query_mass=mass, charge=charge, ppm_error=5.0, max_results=10000,
        rdbe_coeffs=rdbe_coeffs, rdbe_min=-0.5, rdbe_max=40.0, check_octet=True)
    counts_all, _ = finder.decomposer.decompose_to_counts(
        query_mass=mass, charge=charge, ppm_error=5.0, max_results=10000,
        rdbe_coeffs=rdbe_coeffs, rdbe_min=-0.5, rdbe_max=40.0, check_octet=True,
        iso_m1_coeffs=m1_coeffs, iso_m2_direct_coeffs=m2_coeffs,
        obs_m1_ratio=obs_m1, obs_m2_ratio=obs_m2)

    print(f"  Decompose only:           {t_decompose*1e3:6.2f} ms  ({len(counts_raw)} candidates)")
    print(f"  Decompose + RDBE/octet:   {t_decompose_rdbe*1e3:6.2f} ms  ({len(counts_rdbe)} candidates)")
    print(f"  Decompose + RDBE + preF:  {t_decompose_all*1e3:6.2f} ms  ({len(counts_all)} candidates)")

    # 4. Full pipeline with isotope matching
    def step_full_iso():
        iso_config = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
            enable_approx_prefilter=True,
        )
        finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
            isotope_match=iso_config,
        )
    t_full = time_fn(step_full_iso, n=50)

    # 5. Full pipeline without isotope
    def step_full_no_iso():
        finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
    t_full_no_iso = time_fn(step_full_no_iso, n=100)

    iso_config = SingleEnvelopeMatch(
        envelope=envelope.copy(), mz_tolerance_ppm=5.0, minimum_rmse=0.1,
        enable_approx_prefilter=True)
    results_iso = finder.find_formulae(
        mass=mass, charge=charge, error_ppm=5.0,
        filter_rdbe=(-0.5, 40), check_octet=True, isotope_match=iso_config)
    results_no_iso = finder.find_formulae(
        mass=mass, charge=charge, error_ppm=5.0,
        filter_rdbe=(-0.5, 40), check_octet=True)

    print(f"\n  Full pipeline (no iso):   {t_full_no_iso*1e3:6.2f} ms  ({len(results_no_iso)} candidates)")
    print(f"  Full pipeline (with iso): {t_full*1e3:6.2f} ms  ({len(results_iso)} candidates)")
    print(f"\n  Isotope scoring overhead: {(t_full - t_full_no_iso)*1e3:6.2f} ms")
    print(f"  Per-candidate iso cost:   {(t_full - t_full_no_iso) / max(len(counts_rdbe), 1) * 1e6:.1f} µs")

    # 6. Break down the Python overhead in find_formulae
    print(f"\n  --- Python overhead analysis ---")
    # Time just the candidate construction loop
    counts_arr = counts_rdbe
    real_masses_arr = finder.decomposer.real_masses
    symbols = list(finder.decomposer.element_symbols)

    from find_mfs.core.finder import ELECTRON, LightFormula
    charge_offset = ELECTRON.mass * charge
    counts_f = counts_arr.astype(np.float64)
    exact_masses = counts_f @ real_masses_arr - charge_offset
    all_err_ppm = (exact_masses - mass) / mass * 1e6
    all_rdbes = counts_f @ rdbe_coeffs + 1.0
    sort_order = np.argsort(np.abs(all_err_ppm))

    # a) Vectorized mass/error/rdbe computation
    def step_vectorized():
        cf = counts_arr.astype(np.float64)
        em = cf @ real_masses_arr - charge_offset
        errs = (em - mass) / mass * 1e6
        rdbes = cf @ rdbe_coeffs + 1.0
        np.argsort(np.abs(errs))
    t_vectorized = time_fn(step_vectorized)

    # b) LightFormula construction
    sorted_counts_list = counts_arr[sort_order].tolist()
    sorted_masses = exact_masses[sort_order].tolist()
    def step_light_formula():
        for i in range(min(100, len(sorted_counts_list))):
            LightFormula.from_counts(
                symbols=symbols, counts=sorted_counts_list[i],
                charge=charge, monoisotopic_mass=sorted_masses[i],
            )
    t_formula = time_fn(step_light_formula, n=50)

    # c) SingleEnvelopeMatch construction (the copy + normalize)
    def step_config():
        SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.1,
        )
    t_config = time_fn(step_config)

    n_cands = len(sorted_counts_list)
    print(f"  Vectorized mass/err/rdbe: {t_vectorized*1e6:7.0f} µs  ({n_cands} candidates)")
    print(f"  LightFormula x100:       {t_formula*1e6:7.0f} µs  ({t_formula/100*1e6:.1f} µs/each)")
    print(f"  Config construction:     {t_config*1e6:7.0f} µs")


def profile_larger_mass():
    """Profile at 1000 Da where there are many more candidates."""
    print("\n" + "=" * 70)
    print("LARGE MASS PROFILE (1000 Da)")
    print("=" * 70)

    finder = FormulaFinder("CHNOPS")
    mass = 1000.0
    charge = 1

    envelope = np.array([
        [1000.0, 1.0],
        [1001.003, 0.55],
        [1002.007, 0.16],
    ])

    # Count at each stage
    rdbe_coeffs = finder._rdbe_coeffs
    m1_coeffs = finder._iso_m1_coeffs
    m2_coeffs = finder._iso_m2_direct_coeffs

    counts_raw, _ = finder.decomposer.decompose_to_counts(
        query_mass=mass, charge=charge, ppm_error=5.0, max_results=10000)
    counts_rdbe, _ = finder.decomposer.decompose_to_counts(
        query_mass=mass, charge=charge, ppm_error=5.0, max_results=10000,
        rdbe_coeffs=rdbe_coeffs, rdbe_min=-0.5, rdbe_max=40.0, check_octet=True)

    print(f"  Raw decompositions:     {len(counts_raw)}")
    print(f"  After RDBE/octet:       {len(counts_rdbe)}")

    def step_full():
        iso_config = SingleEnvelopeMatch(
            envelope=envelope.copy(),
            mz_tolerance_ppm=5.0,
            minimum_rmse=0.05,
        )
        return finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
            isotope_match=iso_config,
        )
    t_full = time_fn(step_full, n=20)
    results = step_full()
    print(f"  After isotope match:    {len(results)}")
    print(f"  Full pipeline time:     {t_full*1e3:.1f} ms")
    print(f"  Per-candidate cost:     {t_full / max(len(counts_rdbe), 1) * 1e6:.1f} µs")

    # No-iso baseline
    def step_no_iso():
        return finder.find_formulae(
            mass=mass, charge=charge, error_ppm=5.0,
            filter_rdbe=(-0.5, 40), check_octet=True,
        )
    t_no_iso = time_fn(step_no_iso, n=50)
    print(f"  No-iso pipeline time:   {t_no_iso*1e3:.1f} ms")
    print(f"  Iso overhead:           {(t_full - t_no_iso)*1e3:.1f} ms")
    print(f"  Iso overhead per cand:  {(t_full - t_no_iso) / max(len(counts_rdbe), 1) * 1e6:.1f} µs")


def profile_nrt_overhead():
    """Measure the NRT (Numba Runtime) overhead from the codegen report."""
    print("\n" + "=" * 70)
    print("NRT OVERHEAD ANALYSIS")
    print("=" * 70)

    # The codegen showed 27 NRT_decref calls. Let's check if result array
    # slicing (result[:count]) is the cause
    n = 6
    ERT = np.zeros((3, 10), dtype=np.int64)
    integer_masses = np.array([12, 1, 14, 16, 31, 32], dtype=np.int64)
    real_masses = np.array([12.0, 1.00783, 14.003, 15.995, 30.974, 31.972], dtype=np.float64)
    bounds = np.array([100.0, 200.0, 30.0, 30.0, 10.0, 10.0], dtype=np.float64)
    min_values = np.zeros(n, dtype=np.int64)
    rdbe_coeffs = np.zeros(n, dtype=np.float64)
    iso_m1 = np.zeros(n, dtype=np.float64)
    iso_m2 = np.zeros(n, dtype=np.float64)

    # Trigger compilation
    _decompose_mass_range(
        ERT, integer_masses, real_masses, bounds, min_values,
        100, 100, 99.0, 101.0, 0.0, 10, rdbe_coeffs,
        -0.5, 40.0, True, True, True,
        False, iso_m1, iso_m2, 0.0, 0.0, 0.3, 0.02,
    )

    sigs = list(_decompose_mass_range.signatures)
    llvm = _decompose_mass_range.inspect_llvm(sigs[0])

    # Count NRT calls and try to find what they're associated with
    import re
    nrt_lines = [line.strip() for line in llvm.split('\n')
                 if 'NRT_' in line]
    print(f"  Total NRT calls: {len(nrt_lines)}")

    # Count different NRT operations
    nrt_types = {}
    for line in nrt_lines:
        for pattern in ['NRT_incref', 'NRT_decref', 'NRT_MemInfo_alloc_safe',
                        'NRT_MemInfo_call_dtor', 'NRT_Allocate', 'NRT_Free']:
            if pattern in line:
                nrt_types[pattern] = nrt_types.get(pattern, 0) + 1

    for op, count in sorted(nrt_types.items()):
        print(f"    {op}: {count}")

    # Check for array allocations inside the function
    alloc_count = llvm.count('NRT_MemInfo_alloc')
    print(f"\n  Array allocations inside function: {alloc_count}")
    print("  (These are for: result buffer, c array, and return slice)")


def profile_object_creation():
    """Profile Python object creation overhead."""
    print("\n" + "=" * 70)
    print("PYTHON OBJECT CREATION OVERHEAD")
    print("=" * 70)

    from find_mfs.core.finder import LightFormula, FormulaCandidate, ELECTRON
    from find_mfs.isotopes.results import SingleEnvelopeMatchResult

    symbols = ['C', 'H', 'N', 'O', 'P', 'S']
    counts = [31, 36, 2, 11, 0, 0]

    # LightFormula.from_counts
    def step_lf():
        return LightFormula.from_counts(symbols, counts, 1, 612.211)
    t_lf = time_fn(step_lf, n=500)

    # FormulaCandidate creation
    lf = step_lf()
    def step_fc():
        return FormulaCandidate(
            formula=lf, error_ppm=1.23, error_da=0.001, rdbe=15.0)
    t_fc = time_fn(step_fc, n=500)

    # SingleEnvelopeMatchResult creation
    def step_result():
        return SingleEnvelopeMatchResult(
            num_peaks_matched=3, num_peaks_total=3,
            intensity_rmse=0.01, match_fraction=1.0,
            peak_matches=np.full(3, True),
            predicted_envelope=np.empty((0, 2)),
        )
    t_result = time_fn(step_result, n=500)

    # .tolist() conversion
    arr = np.random.randint(0, 100, (500, 6), dtype=np.int32)
    def step_tolist():
        arr.tolist()
    t_tolist = time_fn(step_tolist, n=100)

    print(f"  LightFormula.from_counts:     {t_lf*1e6:6.1f} µs/call")
    print(f"  FormulaCandidate:             {t_fc*1e6:6.1f} µs/call")
    print(f"  SingleEnvelopeMatchResult:    {t_result*1e6:6.1f} µs/call")
    print(f"  np.tolist() for 500x6 array:  {t_tolist*1e6:6.0f} µs")
    print(f"  np.tolist() per row:          {t_tolist/500*1e6:6.2f} µs")


if __name__ == "__main__":
    profile_old_path_breakdown()
    profile_new_path_breakdown()
    profile_pipeline_breakdown()
    profile_larger_mass()
    profile_nrt_overhead()
    profile_object_creation()
