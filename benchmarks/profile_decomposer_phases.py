"""Break down the Python overhead inside decomposer.decompose_and_score."""
import time
import numpy as np
import math
from find_mfs import FormulaFinder

finder = FormulaFinder('CHNOPS')
d = finder.decomposer

n = 5000

# Phase 1: _populate_counts_based_on_user_input
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    max_counts = {s: 10000 for s in d.element_symbols}
    min_counts = {s: 0 for s in d.element_symbols}
    d._populate_counts_based_on_user_input(max_counts, min_counts)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"_populate_counts:    median={np.median(arr):.2f} us")

# Phase 2: _get_possible_element_count_bounds
max_counts = {s: 10000 for s in d.element_symbols}
min_counts = {s: 0 for s in d.element_symbols}
max_counts, min_counts = d._populate_counts_based_on_user_input(max_counts, min_counts)
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    d._get_possible_element_count_bounds(max_counts, min_counts)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"_get_bounds:         median={np.median(arr):.2f} us")

# Phase 3: _populate_mass_range x2
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    d._populate_mass_range_based_on_user_input(mass=180.063388, ppm_error=5.0, mz_error=0.0)
    d._populate_mass_range_based_on_user_input(mass=180.063388, ppm_error=5.0, mz_error=0.0, min_counts=min_counts)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"_mass_range (x2):    median={np.median(arr):.2f} us")

# Phase 4: mass bounds loop
bounds, min_values = d._get_possible_element_count_bounds(max_counts, min_counts)
_, max_mass = d._populate_mass_range_based_on_user_input(mass=180.063388, ppm_error=5.0, mz_error=0.0, min_counts=min_counts)
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    for i, element in enumerate(d.elements):
        mass_bound = math.floor(max_mass / element.mass)
        bounds[i] = min(bounds[i], float(mass_bound))
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"mass_bounds loop:    median={np.median(arr):.2f} us")

# Phase 5: _get_mass_range_as_integers
min_mass = max(0.0, 180.063388 - 180.063388 * 5e-6)
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    d._get_mass_range_as_integers(min_mass=min_mass, max_mass=max_mass)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"_get_mass_range_int: median={np.median(arr):.2f} us")

# Phase 6: np.array creation for bounds/min_values
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    bounds_arr = np.array(bounds, dtype=np.float64)
    min_values_arr = np.array(min_values, dtype=np.int64)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"np.array (bounds):   median={np.median(arr):.2f} us")

# Phase 7: np.zeros + ascontiguousarray for iso/rdbe
times = []
n_elem = len(d.elements)
for _ in range(n):
    t0 = time.perf_counter_ns()
    rdbe_coeffs_arr = np.zeros(n_elem, dtype=np.float64)
    iso_m1_arr = np.zeros(n_elem, dtype=np.float64)
    iso_m2_arr = np.zeros(n_elem, dtype=np.float64)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"np.zeros (3 arrays): median={np.median(arr):.2f} us")

# Phase 8: The Python import overhead
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    from find_mfs.core.algorithms import decompose_and_score as _cds
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"import (cached):     median={np.median(arr):.2f} us")

# Phase 9: find_formulae Python-only overhead (arg parsing, adduct, kwargs building)
# Simulate the find_formulae front-matter
times = []
for _ in range(n):
    t0 = time.perf_counter_ns()
    # adduct parsing
    adduct_formula = None
    adduct_mass = 0.0
    # constraint parsing
    min_counts_dict = None
    max_counts_dict = None
    adjusted_mass = 180.063388 - adduct_mass
    symbols = finder._symbols
    can_prefilter = True
    rdbe_coeffs = finder._rdbe_coeffs
    decompose_kwargs = {
        'rdbe_coeffs': rdbe_coeffs,
        'rdbe_min': -0.5,
        'rdbe_max': 40.0,
        'check_octet': True,
        'charge_parity_even': None,
    }
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / 1e3)
arr = np.array(times)
print(f"find_formulae prep:  median={np.median(arr):.2f} us")
