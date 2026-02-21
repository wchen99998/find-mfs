"""
Inspect Numba-generated LLVM IR and x86 assembly for _decompose_mass_range.

Triggers JIT compilation with representative types, then dumps:
  1. Typed signature info
  2. LLVM IR (highlights hot-path patterns)
  3. Native x86-64 assembly

Usage:
    python benchmarks/inspect_llvm.py [--llvm] [--asm] [--types] [--all]
    python benchmarks/inspect_llvm.py --llvm > llvm_ir.ll
"""
import os
# Disable Numba cache BEFORE importing anything that uses @njit(cache=True)
os.environ['NUMBA_DISABLE_JIT_CACHE'] = '1'

import sys
import numpy as np

from find_mfs.core.algorithms import _decompose_mass_range


def trigger_compilation():
    """Call _decompose_mass_range with representative args to force JIT."""
    n = 6  # CHNOPS
    ERT = np.zeros((100, n), dtype=np.float64)
    ERT[0, :] = 0
    integer_masses = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    real_masses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    bounds = np.array([10.0] * n, dtype=np.float64)
    min_values = np.zeros(n, dtype=np.int64)
    rdbe_coeffs = np.array([-0.5, 1.0, 0.5, 0.0, 1.5, 0.0], dtype=np.float64)

    _decompose_mass_range(
        ERT, integer_masses, real_masses, bounds, min_values,
        0, 1, 0.0, 100.0, 0.0, 100,
        rdbe_coeffs, -1e30, 1e30, False, True, False,
    )


def print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


def show_types():
    print_section("NUMBA TYPE SIGNATURES")
    _decompose_mass_range.inspect_types()


def show_llvm():
    print_section("LLVM IR")
    for sig, ir in _decompose_mass_range.inspect_llvm().items():
        print(f"--- Signature: {sig} ---")
        lines = ir.splitlines()
        print(f"    Total LLVM IR lines: {len(lines)}")
        print()

        # Count key instruction patterns
        patterns = {
            'vfmadd / fmuladd': 0,
            'fadd': 0,
            'fmul': 0,
            'fcmp': 0,
            'br (branch)': 0,
            'phi': 0,
            'load': 0,
            'store': 0,
            'call': 0,
            'NRT_incref': 0,
            'NRT_decref': 0,
            'srem / urem (modulo)': 0,
            'sdiv / udiv': 0,
            'alloca': 0,
            'getelementptr': 0,
        }

        for line in lines:
            stripped = line.strip()
            if 'fmuladd' in stripped or 'vfmadd' in stripped:
                patterns['vfmadd / fmuladd'] += 1
            if ' fadd ' in stripped:
                patterns['fadd'] += 1
            if ' fmul ' in stripped:
                patterns['fmul'] += 1
            if ' fcmp ' in stripped:
                patterns['fcmp'] += 1
            if stripped.startswith('br '):
                patterns['br (branch)'] += 1
            if ' phi ' in stripped:
                patterns['phi'] += 1
            if ' load ' in stripped:
                patterns['load'] += 1
            if ' store ' in stripped:
                patterns['store'] += 1
            if ' call ' in stripped:
                patterns['call'] += 1
            if 'NRT_incref' in stripped:
                patterns['NRT_incref'] += 1
            if 'NRT_decref' in stripped:
                patterns['NRT_decref'] += 1
            if ' srem ' in stripped or ' urem ' in stripped:
                patterns['srem / urem (modulo)'] += 1
            if ' sdiv ' in stripped or ' udiv ' in stripped:
                patterns['sdiv / udiv'] += 1
            if 'alloca' in stripped:
                patterns['alloca'] += 1
            if 'getelementptr' in stripped:
                patterns['getelementptr'] += 1

        print("  Instruction counts:")
        for pat, cnt in patterns.items():
            flag = ""
            if 'NRT' in pat and cnt > 0:
                flag = "  *** WARNING: ref-counting overhead ***"
            if 'fmuladd' in pat and cnt > 0:
                flag = "  (FMA fusion active)"
            print(f"    {pat:30s}: {cnt:5d}{flag}")

        print()
        print("  Full LLVM IR follows:")
        print("  " + "-" * 68)
        print(ir)


def show_asm():
    print_section("NATIVE x86-64 ASSEMBLY")
    for sig, asm in _decompose_mass_range.inspect_asm().items():
        print(f"--- Signature: {sig} ---")
        asm_lines = asm.splitlines()
        print(f"    Total ASM lines: {len(asm_lines)}")
        print()

        # Count x86 patterns of interest
        x86_patterns = {
            'vfmadd (FMA)': 0,
            'vmulsd/vmulss': 0,
            'vaddsd/vaddss': 0,
            'vdivsd': 0,
            'vcmpsd/vucomisd': 0,
            'idiv': 0,
            'imul': 0,
            'jmp/jcc (jumps)': 0,
            'call': 0,
            'NRT_incref': 0,
            'NRT_decref': 0,
            'movsd/movss (fp mov)': 0,
            'mov (integer)': 0,
        }

        for line in asm_lines:
            stripped = line.strip()
            parts = stripped.split()
            if not parts:
                continue
            mnemonic = parts[0]

            if mnemonic.startswith('vfmadd') or mnemonic.startswith('vfnmadd') or mnemonic.startswith('vfmsub'):
                x86_patterns['vfmadd (FMA)'] += 1
            elif mnemonic in ('vmulsd', 'vmulss', 'mulsd', 'mulss'):
                x86_patterns['vmulsd/vmulss'] += 1
            elif mnemonic in ('vaddsd', 'vaddss', 'addsd', 'addss'):
                x86_patterns['vaddsd/vaddss'] += 1
            elif mnemonic in ('vdivsd', 'divsd'):
                x86_patterns['vdivsd'] += 1
            elif mnemonic in ('vcmpsd', 'vucomisd', 'ucomisd', 'cmpsd'):
                x86_patterns['vcmpsd/vucomisd'] += 1
            elif mnemonic == 'idiv' or mnemonic == 'idivq':
                x86_patterns['idiv'] += 1
            elif mnemonic.startswith('imul'):
                x86_patterns['imul'] += 1
            elif mnemonic.startswith('j'):
                x86_patterns['jmp/jcc (jumps)'] += 1
            elif mnemonic == 'callq' or mnemonic == 'call':
                x86_patterns['call'] += 1

            if 'NRT_incref' in stripped:
                x86_patterns['NRT_incref'] += 1
            if 'NRT_decref' in stripped:
                x86_patterns['NRT_decref'] += 1

            if mnemonic in ('movsd', 'movss', 'vmovsd', 'vmovss'):
                x86_patterns['movsd/movss (fp mov)'] += 1
            elif mnemonic.startswith('mov') and not mnemonic.startswith('movs'):
                x86_patterns['mov (integer)'] += 1

        print("  x86 instruction counts:")
        for pat, cnt in x86_patterns.items():
            flag = ""
            if 'NRT' in pat and cnt > 0:
                flag = "  *** WARNING ***"
            if 'FMA' in pat and cnt > 0:
                flag = "  (hardware FMA used)"
            print(f"    {pat:30s}: {cnt:5d}{flag}")

        print()
        print("  Full assembly follows:")
        print("  " + "-" * 68)
        print(asm)


def main():
    args = set(sys.argv[1:])
    if not args:
        args = {'--all'}

    print("Triggering JIT compilation (cache disabled for inspection)...")
    trigger_compilation()
    print("Done.\n")

    if '--types' in args or '--all' in args:
        show_types()

    if '--llvm' in args or '--all' in args:
        show_llvm()

    if '--asm' in args or '--all' in args:
        show_asm()


if __name__ == "__main__":
    main()
