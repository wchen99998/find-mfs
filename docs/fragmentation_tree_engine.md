# Fragmentation Tree Engine

This note records the public `find_mfs` API and the SIRIUS v6 reference call
used for validation.

## Public API

The fragmentation-tree API follows the same shape as the formula finder:
instantiate a finder with an element alphabet, pass explicit inputs or a
spectrum object, and receive a result object with table helpers.

```python
from find_mfs import (
    ExplicitFragmentationScoring,
    FragmentCandidate,
    FragmentationTreeFinder,
    FragmentationTreeOptions,
)

finder = FragmentationTreeFinder("CHNO")
tree = finder.find_tree(
    root_candidates=[
        FragmentCandidate("C4H8", 56.0, score=1.0, peak_id=0, color=0),
    ],
    fragment_candidates=[
        FragmentCandidate("C3H6", 42.0, score=3.0, peak_id=1, color=1),
        FragmentCandidate("C2H4", 28.0, score=5.0, peak_id=2, color=2),
    ],
    scoring=ExplicitFragmentationScoring(
        peak_scores={1: 4.0, 2: 7.0},
        peak_pair_scores={(0, 1): 1.0, (1, 2): 2.0},
        loss_scores={("C4H8", "C3H6"): 8.0, ("C3H6", "C2H4"): 14.0},
    ),
    options=FragmentationTreeOptions(threads=1),
)

print(tree.tree_score)
print(tree.to_table())
print(tree.losses_table())
```

Use explicit candidates when another scorer, parser, or reference engine already
assigned formula candidates and edge weights. The Rust backend solves the exact
colorful-subtree ILP through `good_lp` with HiGHS.

This explicit path is the stable engine boundary for future integrations:

- `FragmentCandidate.score` is the candidate/root decomposition term.
- `ExplicitFragmentationScoring.peak_scores` adds one score per peak color.
- `peak_pair_scores` adds parent-color to child-color terms.
- `fragment_scores` adds formula-level terms independent of the chosen parent.
- `loss_scores` adds parent-formula to child-formula terms.
- `FragmentationTreeOptions` carries solver controls such as graph reduction,
  time limit, thread count, and optional minimum objective.

## Raw Spectrum API

Use `find_tree_from_spectrum()` when the input is raw MS/MS peaks and the
precursor formula is already known.

```python
from find_mfs import (
    FragmentationSpectrum,
    FragmentationTreeFinder,
    SiriusLikeScoringTables,
    SiriusLikeScoringConfig,
    SpectrumPeak,
)

spectrum = FragmentationSpectrum(
    name="Novobiocin",
    precursor_mz=613.2392,
    precursor_formula="C31H36N2O11",
    precursor_ion="[M+H]+",
    peaks=[
        SpectrumPeak(125.0601, 13556),
        SpectrumPeak(143.0719, 10078),
        SpectrumPeak(189.0925, 999999),
        SpectrumPeak(200.0943, 8848),
        SpectrumPeak(218.1051, 8628),
        SpectrumPeak(613.2512, 64161),
    ],
)

tree = FragmentationTreeFinder("CHNO").find_tree_from_spectrum(
    spectrum,
    scoring_config=SiriusLikeScoringConfig(),
)
```

The public raw-spectrum call defaults to the end-to-end Rust implementation:
preprocessing, candidate generation, SIRIUS-like scoring, graph construction,
and the `good_lp`/HiGHS solve all run in Rust. The previous Python orchestration
path remains available for parity diagnostics:

```python
python_tree = FragmentationTreeFinder("CHNO").find_tree_from_spectrum(
    spectrum,
    implementation="python",
)
```

For the lowest-overhead path, keep the result in Rust:

```python
rust_result = FragmentationTreeFinder("CHNO").find_tree_result_from_spectrum(spectrum)
rust_result.formula_strings()
rust_result.losses()
```

The built-in scorer uses static tables embedded in Rust, so the default path
does not rebuild or pass scoring lookup tables for each call. Callers that want
to adjust the SIRIUS-like lookup tables can opt in with
`SiriusLikeScoringTables`; those tables are copied into the Rust finder once,
then reused for every raw-spectrum call through that finder:

```python
tables = SiriusLikeScoringTables.default()
formula = "C12H12O2"
current = tables.common_fragments.get(formula, 0.0)
tables.common_fragments[formula] = current + 5.0

finder = FragmentationTreeFinder("CHNO", scoring_tables=tables)
tree = finder.find_tree_from_spectrum(spectrum)
```

Mutating `tables` after the first call does not change an already cached Rust
finder. Create a new `FragmentationTreeFinder` when a different table profile is
intended. This keeps the normal API fast while still exposing a coherent escape
hatch for caller-owned scoring profiles.

The validation harness `benchmarks/compare_fragmentation_python_rust.py`
compares these two implementations on stratified MassBank samples and writes
JSONL/CSV summaries for future regression checks.

This path first applies SIRIUS-style parent-peak handling: MS2 peaks inside
twice the configured MS2 tolerance around the precursor m/z are treated as the
parent window, using SIRIUS's `Deviation(10)` absolute floor (`0.002 Da`,
doubled for the merge window). By default, lower-intensity peaks inside that
doubled MS2 window are removed before parent handling, matching SIRIUS's
single-spectrum high-intensity merger; callers can set
`SiriusLikeScoringConfig(merge_close_peaks=False)` to preserve all raw peaks
when that is preferable for a non-SIRIUS profile. The nearest parent-window
peak above 10% of the parent-window maximum is used as the root peak, and
parent-window peaks plus peaks within 0.1 Da below the precursor are excluded
from fragment candidate generation. The default cap is applied to the
intensity-ranked pool of fragment peaks plus the real parent peak, so an
intense parent consumes one retained slot while a weak parent can remain
outside the fragment cap. If no parent peak is present, the root is synthetic
and receives zero root mass-deviation penalty. Fragment peaks are decomposed
with the existing Rust formula finder using a relaxed lower RDBE bound of
`-1.5`, kept inside the precursor formula envelope and the SIRIUS MS2 allowed
window, scored with bundled SIRIUS-like default-profile constants, disjoined
across adjacent near-duplicate peaks by keeping each duplicate formula on the
lower-mass-error peak, and then passed to the same exact tree solver. Fragment
peak intensity scores are normalized against the processed-spectrum base peak,
including the parent peak when present, matching SIRIUS preprocessing before
parent removal. Current raw scoring supports `[M+H]+` spectra.

The default raw scorer is designed as an independent SIRIUS-v6-compatible
profile, not as a SIRIUS runtime wrapper or a promise to copy every SIRIUS
choice. It includes the profile-table scorers that can be represented as local
constants, common-loss recombination, common-root-loss fragment scoring, SIRIUS
v6 common-loss recombination edge cases observed in validation, multimere-loss
scoring, fatty-acid-chain loss scoring, the local CHO root scorer, SIRIUS's
`0.5` vertex and edge mass-deviation weights, the `0.002 Da` vertex
mass-deviation absolute floor, and the tree-size retry loop. Candidate
generation uses a `0.003 Da` absolute search padding and then filters generated
formulas back to SIRIUS's final allowed window `max(10 ppm, 0.002 Da)`, so
low-mass fragments inside SIRIUS's absolute window are not lost to
formula-finder grid boundaries. The `-1.5` RDBE lower bound is evidence-backed
from SIRIUS v6 decompositions that include formulas such as `C8H21O6`, while
still excluding more extreme invalid decompositions that SIRIUS does not emit.

SIRIUS `DBPairedScorer` depends on the packaged `bioformulas.bin.gz` resource;
`find_mfs` does not vendor that resource, so the default DB-paired term is
disabled unless a caller intentionally supplies an equivalent score policy.
The SIRIUS v6 `CarbohydrogenFragment` scorer is implemented behind
`SiriusLikeScoringConfig(enable_carbohydrogen_fragment_score=True)`, but is not
enabled by default because SIRIUS applies it together with `DBPairedScorer`.
Turning on only the CHO fragment bonus can move topologies away from the v6
reference when the DB formula map is absent.

The self-contained default keeps the learned common-fragment table from the
profile constants and uses a stricter tree-size retry quality threshold. The
explicit `sirius_v6_reference(...)` profile disables the local common-fragment
table and switches to SIRIUS's `min(processed_merged_peaks - 2, 15)` tree-size
quality threshold, where the processed peak count includes the real or
synthetic parent peak. SIRIUS v6 sometimes applies nonzero common-fragment
terms, but enabling the local table globally currently worsens the 500-record
strict-reference sweep, so this remains a documented profile-resource boundary
rather than a parity hack.

`SiriusLikeScoringConfig(strict_sirius_radical_parity=True)` switches the free
radical scorer to SIRIUS's Java `MolecularFormula.maybeCharged()` parity rule.
The default remains the topology-stable self-contained profile because exact
SIRIUS parity is coupled to SIRIUS's packaged DB-paired formula map.

When a caller has its own DB formula map, it can be supplied without depending
on SIRIUS runtime resources:

```python
tree = FragmentationTreeFinder("CHNO").find_tree_from_spectrum(
    spectrum,
    scoring_config=SiriusLikeScoringConfig.sirius_v6_reference(
        db_paired_formulas=frozenset({"C31H36N2O11", "C22H21NO6"}),
    ),
)
```

For larger caller-owned formula sets, keep the map external and load it at the
API edge:

```python
from find_mfs import load_db_paired_formulas

db_formulas = load_db_paired_formulas("db_formulas.txt")
tree = FragmentationTreeFinder("CHNO").find_tree_from_spectrum(
    spectrum,
    scoring_config=SiriusLikeScoringConfig.sirius_v6_reference(
        db_paired_formulas=db_formulas,
    ),
)
```

That path is the intended public extension point for projects that want closer
SIRIUS score reproduction while keeping `find_mfs` independent of the packaged
SIRIUS `bioformulas.bin.gz` resource.

## SIRIUS v6 Reference Call

SIRIUS v6 CLI `formulas` is the intended command-line analogue for
formula-constrained fragmentation trees, but in this workspace SIRIUS 6.3.6
exited with `Login ERROR: Please Login to use the SIRIUS command line tool!`
even when called with only local formula-tree options. The core Java API did
compute the same local task without login.

The tracked helper `benchmarks/SiriusV6ApiCompute.java` calls that API:

```bash
CP='tmp/sirius-v6/sirius/lib/app/*:tmp/sirius-v6/sirius/lib/app/lib/*'

tmp/sirius-v6/sirius/lib/runtime/bin/java \
  -Djava.library.path=tmp/sirius-v6/sirius/lib/app/lib \
  -classpath "$CP" \
  benchmarks/SiriusV6ApiCompute.java \
  tmp/sirius-v6/refs/novobiocin.ms \
  C31H36N2O11 \
  --scores
```

The helper prints tab-separated `TREE_SCORE`, `ROOT`, `FRAGMENT`, and `LOSS`
records. With `--scores`, it also prints SIRIUS fragment and loss score
annotations. It configures SIRIUS for fixed-formula tree computation:

- CLP tree builder
- no recalibration
- no isotope score/filter
- one candidate and one candidate per ionization
- bottom-up/de novo formula search disabled

Additional tracked helpers are available for future scorer diagnostics:

- `benchmarks/SiriusV6PreprocessPeaks.java` prints the SIRIUS merged peaks,
  parent peak, relative intensities, and original peaks.
- `benchmarks/SiriusV6DumpDecompositions.java` prints processed-peak
  decompositions, and with `--graph` also prints SIRIUS graph fragments and
  loss weights before tree solving.
- `benchmarks/SiriusV6DumpBioFormulas.java` dumps SIRIUS's packaged
  DB-paired formula map so callers can reproduce the validation profile
  without vendoring the SIRIUS resource.
- `benchmarks/SiriusV6FormulaDiagnostics.java` prints SIRIUS doubled-RDBE and
  valence-filter diagnostics for individual formulas.

For larger validation runs, use the MassBank comparison harness:

```bash
uv run python benchmarks/compare_massbank_sirius.py \
  --limit 100 \
  --sirius-home tmp/sirius-v6/sirius \
  --output-dir tmp/massbank_sirius_compare_100
```

For closest v6-reference reproduction, supply a caller-owned DB formula map and
disable the learned common-fragment table:

```bash
uv run python benchmarks/compare_massbank_sirius.py \
  --limit 500 \
  --massbank-dir tmp/MassBank-data \
  --sirius-home tmp/sirius-v6/sirius \
  --strict-sirius-radical-parity \
  --enable-carbohydrogen-fragment-score \
  --disable-common-fragment-score \
  --use-sirius-tree-size-quality-threshold \
  --db-paired-formulas tmp/sirius-v6/bioformulas.txt \
  --output-dir tmp/massbank_sirius_compare_500
```

The script downloads and filters MassBank records, caches SIRIUS `.ms` inputs and
outputs, and writes per-spectrum formula/loss overlap metrics. Use
`--massbank-dir` with a local `MassBank-data` checkout when scaling beyond the
starter set, and add `--shuffle --seed <n> --max-candidates 0` when the sample
should span MassBank source directories instead of using the deterministic
path-sorted prefix.

## License Boundary

The `find_mfs` implementation does not link to or vendor SIRIUS. It uses an
independent Rust ILP implementation and treats SIRIUS v6 output as validation
data.

For SIRIUS itself, keep two separate points in mind:

- The packaged SIRIUS distribution is AGPL-3.0-or-later according to its
  bundled `COPYING.txt` and GitHub repository metadata.
- The Java source files for the `sirius_api` and
  `fragmentation_tree_construction` modules inspected here carry
  LGPL-3.0-or-later headers, but that is a source-code licensing fact, not a
  statement that redistributing the full SIRIUS application has no obligations.

SIRIUS v6 documentation says user account/licence requirements apply to web
service features such as CSI:FingerID, CANOPUS, and MSNovelist, while local
fragmentation-tree computation ships with a COIN-OR solver. In this repository,
avoid depending on SIRIUS runtime code; use the direct API helper only to
regenerate reference outputs.

Relevant upstream pages:

- <https://v6.docs.sirius-ms.io/install/#user-account-and-license-since-v500>
- <https://v6.docs.sirius-ms.io/quick-start/#example-2-ms-files>
- <https://github.com/sirius-ms/sirius>
- <https://github.com/rust-or/good_lp>

`good_lp` is MIT-licensed and models MILP problems while delegating solving to
feature-selected solver crates. Its HiGHS feature supports integer variables
and statically links HiGHS on Linux, with a C/C++ compiler still required at
build time.
