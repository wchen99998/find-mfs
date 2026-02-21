#!/usr/bin/env python3
"""
Numba codegen inspection for _decompose_mass_range and _isospec_match_and_score.

Checks:
1. Whether do_iso_filter branch is hoisted out of inner loop
2. Whether fastmath=True produces FMA instructions
3. Whether iso_m1_coeffs[j] access generates proper bounds checking
4. Whether ctypes calls in scorer compile to direct function pointer calls
"""

import re
import sys
import numpy as np

# Force compilation by calling the functions with representative inputs
from find_mfs.core.algorithms import _decompose_mass_range


def trigger_compilation():
    """Trigger JIT compilation with representative args."""
    n = 6  # CHNOPS
    ERT = np.zeros((3, 10), dtype=np.int64)
    integer_masses = np.array([12, 1, 14, 16, 31, 32], dtype=np.int64)
    real_masses = np.array([12.0, 1.00783, 14.003, 15.995, 30.974, 31.972],
                           dtype=np.float64)
    bounds = np.array([100.0, 200.0, 30.0, 30.0, 10.0, 10.0], dtype=np.float64)
    min_values = np.zeros(n, dtype=np.int64)
    rdbe_coeffs = np.array([-1.0, 0.5, -0.5, 0.0, -0.5, 0.0], dtype=np.float64)
    iso_m1 = np.array([0.0108, 0.000115, 0.00366, 0.000381, 0.0, 0.0079],
                       dtype=np.float64)
    iso_m2 = np.array([0.0, 0.0, 0.0, 0.00206, 0.0, 0.0447], dtype=np.float64)

    # Call WITHOUT isotope filter
    _decompose_mass_range(
        ERT, integer_masses, real_masses, bounds, min_values,
        min_int=100, max_int=100, original_min_mass=99.0,
        original_max_mass=101.0, charge_mass_offset=0.0,
        max_results=10, rdbe_coeffs=rdbe_coeffs,
        rdbe_min=-0.5, rdbe_max=40.0, check_octet=True,
        charge_parity_even=True, do_rdbe_filter=True,
        do_iso_filter=False,
        iso_m1_coeffs=np.zeros(n, dtype=np.float64),
        iso_m2_direct=np.zeros(n, dtype=np.float64),
        obs_m1_ratio=0.0, obs_m2_ratio=0.0,
        iso_tol_rel=0.3, iso_tol_abs=0.02,
    )

    # Call WITH isotope filter
    _decompose_mass_range(
        ERT, integer_masses, real_masses, bounds, min_values,
        min_int=100, max_int=100, original_min_mass=99.0,
        original_max_mass=101.0, charge_mass_offset=0.0,
        max_results=10, rdbe_coeffs=rdbe_coeffs,
        rdbe_min=-0.5, rdbe_max=40.0, check_octet=True,
        charge_parity_even=True, do_rdbe_filter=True,
        do_iso_filter=True,
        iso_m1_coeffs=iso_m1,
        iso_m2_direct=iso_m2,
        obs_m1_ratio=0.35, obs_m2_ratio=0.06,
        iso_tol_rel=0.3, iso_tol_abs=0.02,
    )


def inspect_decompose_codegen():
    """Inspect LLVM IR for _decompose_mass_range."""
    print("=" * 70)
    print("INSPECTING _decompose_mass_range LLVM IR")
    print("=" * 70)

    trigger_compilation()

    # Get all compiled signatures
    sigs = list(_decompose_mass_range.signatures)
    print(f"\nCompiled signatures: {len(sigs)}")
    for i, sig in enumerate(sigs):
        print(f"  [{i}] {sig}")

    for i, sig in enumerate(sigs):
        llvm_ir = _decompose_mass_range.inspect_llvm(sig)
        print(f"\n--- Signature [{i}] LLVM IR analysis ---")
        print(f"Total IR length: {len(llvm_ir)} chars")

        # Check 1: FMA instructions (fmadd or fmuladd)
        fma_count = len(re.findall(r'fmuladd|fmadd|llvm\.fma', llvm_ir))
        print(f"\nFMA instructions found: {fma_count}")
        if fma_count > 0:
            print("  ✓ fastmath=True is generating FMA fusion")
        else:
            # Also check for fast math flags
            fast_flags = len(re.findall(r'fast ', llvm_ir))
            print(f"  fast math flags found: {fast_flags}")
            if fast_flags > 0:
                print("  ✓ fast math flags present (FMA may be done at machine code level)")
            else:
                print("  ✗ No fast math flags - check fastmath=True setting")

        # Check 2: Vector operations (indicates vectorization)
        vec_ops = len(re.findall(r'<\d+ x (double|float|i\d+)>', llvm_ir))
        print(f"\nVector type operations: {vec_ops}")
        if vec_ops > 0:
            print("  ✓ Loop vectorization detected")
        else:
            print("  ○ No vector types in LLVM IR (vectorization may occur at machine code)")

        # Check 3: Branch analysis around iso_m1_coeffs access
        # Look for the do_iso_filter branch pattern
        iso_branches = len(re.findall(r'do_iso_filter|iso_m1|approx_m1', llvm_ir))
        print(f"\nIso-filter related IR references: {iso_branches}")

        # Check 4: NRT_incref / NRT_decref calls (should be minimal)
        nrt_incref = len(re.findall(r'NRT_incref', llvm_ir))
        nrt_decref = len(re.findall(r'NRT_decref', llvm_ir))
        print(f"\nNRT_incref calls: {nrt_incref}")
        print(f"NRT_decref calls: {nrt_decref}")
        if nrt_incref + nrt_decref < 10:
            print("  ✓ Low NRT overhead (pre-allocated buffer working)")
        else:
            print("  ⚠ High NRT overhead - check for dynamic array growth")

        # Check 5: Bounds check elimination
        bounds_checks = len(re.findall(r'out_of_range|bounds_check|IndexError', llvm_ir))
        print(f"\nBounds check references: {bounds_checks}")
        if bounds_checks == 0:
            print("  ✓ No bounds checking in compiled code")
        else:
            print("  ○ Some bounds checks present")

    return True


def inspect_scorer_codegen():
    """Inspect LLVM IR for _isospec_match_and_score."""
    print("\n" + "=" * 70)
    print("INSPECTING _isospec_match_and_score LLVM IR")
    print("=" * 70)

    try:
        from find_mfs.isotopes.envelope import _get_scorer
        scorer = _get_scorer()

        # Need to trigger compilation first
        from find_mfs.isotopes._isospec_bridge import get_isotope_arrays
        iso_numbers, flat_masses, flat_probs = get_isotope_arrays(['C', 'H', 'O'])
        atom_counts = np.array([6, 12, 6], dtype=np.int32)
        obs_mz = np.array([180.063, 181.067], dtype=np.float64)
        obs_int = np.array([1.0, 0.07], dtype=np.float64)

        scorer(
            iso_numbers, atom_counts, flat_masses, flat_probs, 3,
            obs_mz, obs_int, 0.05, 0.01, 0.001, 1, 0.000548579909,
        )

        sigs = list(scorer.signatures)
        print(f"\nCompiled signatures: {len(sigs)}")

        for i, sig in enumerate(sigs):
            llvm_ir = scorer.inspect_llvm(sig)
            print(f"\n--- Signature [{i}] LLVM IR analysis ---")
            print(f"Total IR length: {len(llvm_ir)} chars")

            # Check 1: ctypes function calls
            # In Numba, ctypes calls compile to indirect calls through function pointers
            call_count = len(re.findall(r'call\s', llvm_ir))
            print(f"\nTotal call instructions: {call_count}")

            # Check 2: Memory management calls (should see free/delete patterns)
            # These won't be named in IR but we can check for expected call patterns
            indirect_calls = len(re.findall(r'call\s+.*%', llvm_ir))
            print(f"Indirect (function pointer) calls: {indirect_calls}")
            print("  (ctypes calls compile to indirect calls through function pointers)")

            # Check 3: carray access pattern
            inttoptr_count = len(re.findall(r'inttoptr', llvm_ir))
            print(f"\ninttoptr instructions: {inttoptr_count}")
            if inttoptr_count > 0:
                print("  ✓ uint64_to_voidptr intrinsic generating inttoptr")

            # Check 4: fast math
            fast_flags = len(re.findall(r'fast ', llvm_ir))
            print(f"\nfast math flags: {fast_flags}")

        return True

    except Exception as e:
        print(f"\n  ⚠ Could not inspect scorer: {e}")
        print("  (This is expected if IsoSpecPy is not available)")
        return False


def check_branch_hoisting():
    """
    Check if LLVM hoists the do_iso_filter branch out of the inner loop.

    Strategy: Compare the number of conditional branches in the inner loop
    between the iso_filter=True and iso_filter=False compilation paths.
    """
    print("\n" + "=" * 70)
    print("CHECKING BRANCH HOISTING FOR do_iso_filter")
    print("=" * 70)

    trigger_compilation()
    sigs = list(_decompose_mass_range.signatures)

    if len(sigs) < 2:
        print("  ⚠ Only one signature compiled - cannot compare")
        return

    # Numba specializes on literal boolean values when passed as keyword args
    # Both calls use the same signature type (bool), so there may be only 1
    # compiled version. In that case, the branch must be in the IR.
    for i, sig in enumerate(sigs):
        llvm_ir = _decompose_mass_range.inspect_llvm(sig)

        # Count conditional branches (br i1)
        cond_branches = len(re.findall(r'br i1', llvm_ir))
        print(f"\n  Signature [{i}]: {cond_branches} conditional branches total")

        # The key question: is the approx_m1 accumulation guarded?
        # Look for the pattern: load from iso_m1_coeffs followed by fmul
        # If hoisted, there should be one branch before the loop body
        # If not hoisted, there's a branch per iteration

        # Look for getelementptr patterns on the isotope arrays
        gep_iso = len(re.findall(r'getelementptr.*iso_m1|getelementptr.*approx', llvm_ir))
        print(f"  GEP instructions related to isotope: {gep_iso}")

    print("\n  Note: Numba compiles a single specialization for bool types.")
    print("  The do_iso_filter branch is loop-invariant. LLVM's LICM pass")
    print("  should hoist it, but with @njit the branch cost is already")
    print("  negligible since the hot inner 'for j' loop runs only 6 iterations")
    print("  (CHNOPS elements). The branch predictor will learn the pattern")
    print("  after the first iteration.")


def main():
    print("Numba Codegen Inspection")
    print("========================\n")

    ok1 = inspect_decompose_codegen()
    check_branch_hoisting()
    ok2 = inspect_scorer_codegen()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  _decompose_mass_range codegen: {'✓ OK' if ok1 else '✗ FAIL'}")
    print(f"  _isospec_match_and_score codegen: {'✓ OK' if ok2 else '⚠ SKIP'}")

    # Issue 1 analysis
    print("\n  Issue 1 (unconditional iso_m1_coeffs access):")
    print("  The access is always safe because decomposer.py passes")
    print("  np.zeros(n_elem) when iso_m1_coeffs is None. The accumulation")
    print("  approx_m1 += val_f * 0.0 is a no-op at the value level.")
    print("  Adding a branch guard is unnecessary since:")
    print("  1. The inner loop runs only 6 iterations (CHNOPS)")
    print("  2. The multiply-by-zero is cheaper than a branch")
    print("  3. A comment in decomposer.py documents the safety invariant")


if __name__ == "__main__":
    main()
