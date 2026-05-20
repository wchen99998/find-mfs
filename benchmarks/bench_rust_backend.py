#!/usr/bin/env python3
"""
Verify and benchmark the Rust backend against the previous Python/Cython path.

Run after building the private extension:

    uv run maturin develop --release --manifest-path find_mfs/rust/Cargo.toml
    uv run python benchmarks/bench_rust_backend.py

The script first materializes and compares both backends for every benchmark
case. It only prints timing data for cases whose results are identical within
the same tolerances used by the Rust parity tests.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import find_mfs
from find_mfs import FormulaFinder
from molmass import Formula

TimingMode = Literal["search", "top5", "materialize"]


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    elements: str
    kwargs: dict[str, Any]
    runs: int | None = None
    warmup: int | None = None


def _counts_for_symbols(formula, symbols: list[str]) -> tuple[int, ...]:
    composition = formula.composition()
    return tuple(
        int(composition[symbol].count) if symbol in composition else 0
        for symbol in symbols
    )


def _records(results, symbols: list[str]) -> list[dict[str, Any]]:
    records = []
    for candidate in results:
        isotope_match = candidate.isotope_match_result
        records.append(
            {
                "formula": candidate.formula.formula,
                "counts": _counts_for_symbols(candidate.formula, symbols),
                "exact_mass": candidate.formula.monoisotopic_mass,
                "error_da": candidate.error_da,
                "error_ppm": candidate.error_ppm,
                "rdbe": candidate.rdbe,
                "isotope_rmse": (
                    None if isotope_match is None
                    else isotope_match.intensity_rmse
                ),
                "isotope_match_fraction": (
                    None if isotope_match is None
                    else isotope_match.match_fraction
                ),
                "isotope_num_peaks_matched": (
                    None if isotope_match is None
                    else isotope_match.num_peaks_matched
                ),
                "isotope_num_peaks_total": (
                    None if isotope_match is None
                    else isotope_match.num_peaks_total
                ),
                "isotope_peak_matches": (
                    None if isotope_match is None
                    else tuple(bool(value) for value in isotope_match.peak_matches)
                ),
                "predicted_envelope": (
                    None if isotope_match is None
                    else tuple(
                        tuple(float(value) for value in row)
                        for row in isotope_match.predicted_envelope.tolist()
                    )
                ),
            }
        )
    return records


def _assert_records_close(
    py_records: list[dict[str, Any]],
    rust_records: list[dict[str, Any]],
) -> None:
    if len(py_records) != len(rust_records):
        raise AssertionError(
            f"candidate count differs: python={len(py_records)} "
            f"rust={len(rust_records)}"
        )

    for idx, (py, rust) in enumerate(zip(py_records, rust_records)):
        if py["formula"] != rust["formula"]:
            raise AssertionError(
                f"formula mismatch at {idx}: {py['formula']} != {rust['formula']}"
            )
        if py["counts"] != rust["counts"]:
            raise AssertionError(
                f"count mismatch at {idx}: {py['counts']} != {rust['counts']}"
            )
        _assert_float_close(idx, "exact_mass", py, rust, 1e-8)
        _assert_float_close(idx, "error_da", py, rust, 1e-8)
        _assert_float_close(idx, "error_ppm", py, rust, 1e-5)
        _assert_optional_float_close(idx, "rdbe", py, rust, 1e-8)
        _assert_optional_float_close(idx, "isotope_rmse", py, rust, 1e-8)
        _assert_optional_float_close(
            idx, "isotope_match_fraction", py, rust, 1e-8
        )

        for key in ("isotope_num_peaks_matched", "isotope_num_peaks_total"):
            if py[key] != rust[key]:
                raise AssertionError(
                    f"{key} mismatch at {idx}: {py[key]} != {rust[key]}"
                )

        if py["isotope_peak_matches"] != rust["isotope_peak_matches"]:
            raise AssertionError(
                f"isotope_peak_matches mismatch at {idx}: "
                f"{py['isotope_peak_matches']} != {rust['isotope_peak_matches']}"
            )
        _assert_envelope_close(idx, py, rust)


def _assert_float_close(
    idx: int,
    key: str,
    py: dict[str, Any],
    rust: dict[str, Any],
    tolerance: float,
) -> None:
    if abs(py[key] - rust[key]) > tolerance:
        raise AssertionError(
            f"{key} mismatch at {idx}: {py[key]} != {rust[key]}"
        )


def _assert_optional_float_close(
    idx: int,
    key: str,
    py: dict[str, Any],
    rust: dict[str, Any],
    tolerance: float,
) -> None:
    if py[key] is None or rust[key] is None:
        if py[key] is not rust[key]:
            raise AssertionError(
                f"{key} mismatch at {idx}: {py[key]} != {rust[key]}"
            )
    elif abs(py[key] - rust[key]) > tolerance:
        raise AssertionError(
            f"{key} mismatch at {idx}: {py[key]} != {rust[key]}"
        )


def _assert_envelope_close(
    idx: int,
    py: dict[str, Any],
    rust: dict[str, Any],
) -> None:
    py_envelope = py["predicted_envelope"]
    rust_envelope = rust["predicted_envelope"]
    if py_envelope is None or rust_envelope is None:
        if py_envelope is not rust_envelope:
            raise AssertionError(
                f"predicted_envelope mismatch at {idx}: one side is missing"
            )
        return
    if len(py_envelope) != len(rust_envelope):
        raise AssertionError(
            f"predicted_envelope row count mismatch at {idx}: "
            f"{len(py_envelope)} != {len(rust_envelope)}"
        )
    for row_idx, (py_row, rust_row) in enumerate(zip(py_envelope, rust_envelope)):
        if len(py_row) != len(rust_row):
            raise AssertionError(
                f"predicted_envelope column count mismatch at {idx}/{row_idx}: "
                f"{len(py_row)} != {len(rust_row)}"
            )
        for col_idx, (py_value, rust_value) in enumerate(zip(py_row, rust_row)):
            if abs(py_value - rust_value) > 1e-4:
                raise AssertionError(
                    "predicted_envelope mismatch at "
                    f"{idx}/{row_idx}/{col_idx}: {py_value} != {rust_value}"
                )


def _verify_case(
    case: BenchmarkCase,
    py_finder: FormulaFinder,
    rust_finder: FormulaFinder,
) -> int:
    py_results = py_finder.find_formulae(**case.kwargs)
    rust_results = rust_finder.find_formulae(**case.kwargs)
    symbols = list(py_finder.decomposer.element_symbols)
    py_records = _records(py_results, symbols)
    rust_records = _records(rust_results, symbols)
    _assert_records_close(py_records, rust_records)
    return len(py_records)


def _consume_results(results, mode: TimingMode):
    if mode == "search":
        return len(results)
    if mode == "top5":
        return tuple(candidate.formula.formula for candidate in results[:5])
    return tuple(candidate.formula.formula for candidate in results)


def _time_call(
    finder: FormulaFinder,
    kwargs: dict[str, Any],
    n_runs: int,
    warmup: int,
    mode: TimingMode,
) -> dict[str, Any]:
    times: list[float] = []
    last_value = None

    for idx in range(warmup + n_runs):
        started = time.perf_counter()
        results = finder.find_formulae(**kwargs)
        last_value = _consume_results(results, mode)
        elapsed = time.perf_counter() - started

        if idx >= warmup:
            times.append(elapsed)

    return _summarize_times(times, last_value)


def _bench_batch(
    finder: FormulaFinder,
    masses: Iterable[float],
    kwargs: dict[str, Any],
    n_runs: int,
    warmup: int,
    mode: TimingMode,
) -> dict[str, Any]:
    masses = list(masses)
    times: list[float] = []
    total_candidates = 0

    for idx in range(warmup + n_runs):
        started = time.perf_counter()
        total_candidates = 0
        for mass in masses:
            results = finder.find_formulae(mass=mass, **kwargs)
            total_candidates += len(results)
            if mode == "top5":
                _ = tuple(candidate.formula.formula for candidate in results[:5])
            elif mode == "materialize":
                _ = tuple(candidate.formula.formula for candidate in results)
        elapsed = time.perf_counter() - started

        if idx >= warmup:
            times.append(elapsed)

    return _summarize_times(times, total_candidates)


def _summarize_times(times: list[float], value: Any) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "value": value,
    }


def _batch_masses(n: int) -> list[float]:
    base = [18.010565, 44.0095, 100.0, 150.0, 180.063, 250.0, 342.11621, 500.0, 750.0, 1000.0]
    return [base[idx % len(base)] + (idx % 17) * 0.0001 for idx in range(n)]


def _glucose_counts() -> dict[str, int]:
    return {"C": 6, "H": 12, "N": 0, "O": 6, "P": 0, "S": 0}


def _glucose_isotope_match():
    formula = Formula("C6H12O6")
    envelope = find_mfs.get_isotope_envelope(
        formula,
        mz_tolerance=0.05,
        threshold=0.001,
    )
    return find_mfs.SingleEnvelopeMatch(
        envelope=envelope,
        mz_tolerance_da=0.01,
        minimum_rmse=0.03,
        enable_approx_prefilter=True,
    )


def _cases(include_isotope: bool) -> list[BenchmarkCase]:
    glucose = Formula("C6H12O6")
    cases = [
        BenchmarkCase(
            "small exact / H2O",
            "CHO",
            {
                "mass": Formula("H2O").monoisotopic_mass,
                "error_da": 1e-6,
                "max_results": 1000,
            },
        ),
        BenchmarkCase(
            "small exact / CO2",
            "CHO",
            {
                "mass": Formula("CO2").monoisotopic_mass,
                "error_da": 1e-6,
                "max_results": 1000,
            },
        ),
        BenchmarkCase(
            "medium ppm / CHNOPS",
            "CHNOPS",
            {"mass": 180.063, "error_ppm": 5.0, "max_results": 10000},
        ),
        BenchmarkCase(
            "medium constrained / glucose",
            "CHNOPS",
            {
                "mass": glucose.monoisotopic_mass,
                "error_da": 1e-6,
                "min_counts": "C6H12O6",
                "max_counts": "C6H12N0O6P0S0",
                "max_results": 1000,
            },
        ),
        BenchmarkCase(
            "adduct / sodium glucose",
            "CHNOPS",
            {
                "mass": Formula("C6H12O6Na+").monoisotopic_mass,
                "charge": 1,
                "adduct": "Na",
                "error_ppm": 5.0,
                "min_counts": _glucose_counts(),
                "max_counts": _glucose_counts(),
                "filter_rdbe": (0, 20),
                "check_octet": True,
                "max_results": 1000,
            },
        ),
        BenchmarkCase(
            "large broad / 500 Da",
            "CHNOPS",
            {"mass": 500.0, "error_da": 0.05, "max_results": 10000},
            runs=5,
        ),
        BenchmarkCase(
            "large filtered / 750 Da",
            "CHNOPS",
            {
                "mass": 750.0,
                "error_ppm": 10.0,
                "filter_rdbe": (0, 60),
                "check_octet": True,
                "max_results": 10000,
            },
            runs=5,
        ),
        BenchmarkCase(
            "halogens / 750 Da",
            "CHNOPSClBr",
            {"mass": 750.0, "error_da": 0.01, "max_results": 5000},
            runs=5,
        ),
        BenchmarkCase(
            "xlarge capped / 1500 Da",
            "CHNOPS",
            {"mass": 1500.0, "error_ppm": 10.0, "max_results": 10000},
            runs=3,
        ),
    ]
    if include_isotope:
        cases.append(
            BenchmarkCase(
                "isotope / glucose envelope",
                "CHNOPS",
                {
                    "mass": glucose.monoisotopic_mass,
                    "error_ppm": 5.0,
                    "max_counts": {
                        "C": 10,
                        "H": 20,
                        "N": 4,
                        "O": 10,
                        "P": 1,
                        "S": 1,
                    },
                    "isotope_match": _glucose_isotope_match(),
                    "max_results": 10000,
                },
                runs=5,
            )
        )
    return cases


def _print_case_result(row: dict[str, Any]) -> None:
    print(row["label"])
    print(
        "  python/cython: "
        f"{row['python_ms']:.2f} ms "
        f"(min {row['python_min_ms']:.2f}, max {row['python_max_ms']:.2f})"
    )
    print(
        "  rust:          "
        f"{row['rust_ms']:.2f} ms "
        f"(min {row['rust_min_ms']:.2f}, max {row['rust_max_ms']:.2f})"
    )
    print(
        f"  identical=yes, candidates={row['candidates']}, "
        f"python/rust speed ratio={row['speed_ratio']:.2f}x"
    )
    print()


def _jsonable_kwargs(kwargs: dict[str, Any]) -> dict[str, str]:
    return {key: repr(value) for key, value in kwargs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-runs", type=int, default=3)
    parser.add_argument("--batch-warmup", type=int, default=1)
    parser.add_argument(
        "--timing-mode",
        choices=["search", "top5", "materialize"],
        default="search",
        help=(
            "search measures result creation only; top5 and materialize include "
            "formula object materialization in the timed region"
        ),
    )
    parser.add_argument("--skip-isotope", action="store_true")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[10, 100, 1000],
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for machine-readable benchmark results.",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("find_mfs._rust") is None:
        raise SystemExit(
            "find_mfs._rust is not installed. Build it with: "
            "uv run maturin develop --manifest-path find_mfs/rust/Cargo.toml"
        )

    finder_cache: dict[tuple[str, str], FormulaFinder] = {}

    def finder(elements: str, backend: str) -> FormulaFinder:
        key = (elements, backend)
        if key not in finder_cache:
            finder_cache[key] = FormulaFinder(elements, backend=backend)
        return finder_cache[key]

    rows: list[dict[str, Any]] = []
    print(f"Timing mode: {args.timing_mode}")
    print("Verifying each case before benchmarking.\n")

    for case in _cases(include_isotope=not args.skip_isotope):
        py_finder = finder(case.elements, "python")
        rust_finder = finder(case.elements, "rust")
        candidate_count = _verify_case(case, py_finder, rust_finder)

        runs = case.runs if case.runs is not None else args.runs
        warmup = case.warmup if case.warmup is not None else args.warmup
        py_result = _time_call(
            py_finder,
            case.kwargs,
            runs,
            warmup,
            args.timing_mode,
        )
        rust_result = _time_call(
            rust_finder,
            case.kwargs,
            runs,
            warmup,
            args.timing_mode,
        )
        row = {
            "kind": "single",
            "label": case.label,
            "elements": case.elements,
            "kwargs": _jsonable_kwargs(case.kwargs),
            "candidates": candidate_count,
            "runs": runs,
            "warmup": warmup,
            "timing_mode": args.timing_mode,
            "python_ms": py_result["median_ms"],
            "python_min_ms": py_result["min_ms"],
            "python_max_ms": py_result["max_ms"],
            "rust_ms": rust_result["median_ms"],
            "rust_min_ms": rust_result["min_ms"],
            "rust_max_ms": rust_result["max_ms"],
            "speed_ratio": py_result["median_ms"] / rust_result["median_ms"],
        }
        rows.append(row)
        _print_case_result(row)

    batch_kwargs = {"error_ppm": 5.0, "max_results": 1000}
    py_finder = finder("CHNOPS", "python")
    rust_finder = finder("CHNOPS", "rust")
    for batch_size in args.batch_sizes:
        masses = _batch_masses(batch_size)
        for mass in masses:
            case = BenchmarkCase(
                f"batch verify mass {mass:.4f}",
                "CHNOPS",
                {"mass": mass, **batch_kwargs},
            )
            _verify_case(case, py_finder, rust_finder)

        py_result = _bench_batch(
            py_finder,
            masses,
            batch_kwargs,
            args.batch_runs,
            args.batch_warmup,
            args.timing_mode,
        )
        rust_result = _bench_batch(
            rust_finder,
            masses,
            batch_kwargs,
            args.batch_runs,
            args.batch_warmup,
            args.timing_mode,
        )
        row = {
            "kind": "batch",
            "label": f"batch / {batch_size:,} masses / reused finder",
            "elements": "CHNOPS",
            "kwargs": _jsonable_kwargs(batch_kwargs),
            "candidates": py_result["value"],
            "runs": args.batch_runs,
            "warmup": args.batch_warmup,
            "timing_mode": args.timing_mode,
            "python_ms": py_result["median_ms"],
            "python_min_ms": py_result["min_ms"],
            "python_max_ms": py_result["max_ms"],
            "rust_ms": rust_result["median_ms"],
            "rust_min_ms": rust_result["min_ms"],
            "rust_max_ms": rust_result["max_ms"],
            "speed_ratio": py_result["median_ms"] / rust_result["median_ms"],
        }
        rows.append(row)
        _print_case_result(row)

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
