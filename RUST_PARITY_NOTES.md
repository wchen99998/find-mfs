# Rust Parity Notes

The Rust backend is intentionally opt-in and mirrors the current
Python/Cython decomposition path. The default `FormulaFinder` backend remains
`"python"` until parity coverage is broad enough to promote Rust.

For `backend="rust"`, the Rust extension now owns the core query pipeline:

- Bocker-Liptak decomposition and scoring
- high-level Rust backend query preparation, including public mass/adduct
  adjustment and user count-bound normalization
- element-set formula string parsing for Rust finder construction
- Python count-bound object extraction for Rust queries (`None`, formula
  strings, and dicts)
- Python isotope-match object extraction for Rust queries, including observed
  envelope rows and tolerance settings
- residual RDBE/octet filtering
- formula-bound string parsing for `min_counts`/`max_counts`
- Hill-ordered formula string formatting for Rust search-result
  materialization
- signed adduct parsing and adduct mass/count offsets
- RDBE coefficient selection, unknown-element residual filter setup, and
  charge-parity selection for RDBE/octet filtering
- approximate isotope prefilter ratio extraction from observed envelopes
- signed-adduct ion count construction
- per-query IsoSpec isotope symbol ordering and flat isotope array assembly
- isotope RMSE filtering and per-peak match metadata
- lazy materialization of predicted isotope envelopes for Rust search results,
  using the same direct IsoSpec C-FFI path instead of Python envelope helpers
- adduct-element dict extraction for Rust lazy isotope-envelope materialization
- Rust-query adduct metadata is preserved as Rust-returned symbol/count vectors
  through lazy envelope materialization instead of being rebuilt as Python dicts
- embedded Rust-owned molmass and IsoSpec isotope/setup tables for finder
  construction
- Rust-owned raw query result storage for `backend="rust"`, including lazy row
  extraction plus common sort/filter operations without materializing Python
  candidate objects
- post-hoc octet filtering on Rust-owned lazy results
- prior/posterior scoring for Rust-owned lazy results using KDE payloads
  extracted from the fitted Python `FormulaPrior`
- Rust-owned display/export rows for `repr()`, `to_table()`, and
  `to_dataframe()` while the result remains unmaterialized, including prior
  scores when present
- the exposed PyO3 query surface is now the Rust-owned finder/result path; the
  earlier Python-prepared raw tuple bridge has been removed from the backend
- element-set-specific finder state, including Rust-built ERT and mass arrays

Python still supplies the IsoSpec shared-library path, then materializes the
existing Python result objects on demand from Rust-owned counts, scores,
isotope metadata, and formula strings.
For `backend="rust"`, Python `MassDecomposer` construction is skipped; Rust
derives most-abundant masses and approximate isotope coefficients from the
embedded source records, sorts element masses, discretizes them, builds the
ERT/error bounds, and stores the resulting finder state once for reuse across
queries.
This keeps the public API stable while reducing the Rust backend's dependence
on Python/Cython execution paths.

## Build

Build the private extension during development with:

```bash
uv run maturin develop --manifest-path find_mfs/rust/Cargo.toml
```

The Python package continues to import and run without `find_mfs._rust`.

Run native Rust tests without PyO3 extension-module linking:

```bash
cargo test --manifest-path find_mfs/rust/Cargo.toml --no-default-features
```

Run parity tests with the extension installed:

```bash
uv run pytest tests/test_rust_parity.py
```

## Known Differences

No intentional candidate-set, ordering, mass-error, RDBE, octet, isotope, or
constraint differences are currently documented. Any future intentional
difference should be added here with a dedicated regression test that
demonstrates the behavior.
