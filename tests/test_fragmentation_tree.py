from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from find_mfs.fragmentation import (
    ExplicitFragmentationScoring,
    Fragment,
    FragmentCandidate,
    FragmentationSpectrum,
    FragmentationTree,
    FragmentationTreeFinder,
    SiriusLikeScoringConfig,
    SpectrumPeak,
    load_db_paired_formulas,
)
from find_mfs.fragmentation.finder import _GeneratedCandidate
from find_mfs.fragmentation.scoring import SiriusLikeScorer

rust = pytest.importorskip("find_mfs._rust")


def test_public_fragmentation_api_is_exported_from_package():
    from find_mfs import (
        ExplicitFragmentationScoring as TopLevelScoring,
        FragmentCandidate as TopLevelFragmentCandidate,
        FragmentationTreeFinder as TopLevelFragmentationTreeFinder,
        SiriusLikeScoringConfig as TopLevelSiriusLikeScoringConfig,
        load_db_paired_formulas as top_level_load_db_paired_formulas,
    )

    assert TopLevelFragmentCandidate is FragmentCandidate
    assert TopLevelFragmentationTreeFinder is FragmentationTreeFinder
    assert TopLevelScoring is ExplicitFragmentationScoring
    assert TopLevelSiriusLikeScoringConfig is SiriusLikeScoringConfig
    assert top_level_load_db_paired_formulas is load_db_paired_formulas


def test_sirius_like_db_paired_score_requires_explicit_formula_set():
    scorer = SiriusLikeScorer("CH", SiriusLikeScoringConfig())
    assert scorer.db_paired_score("C3H6", "C4H8") == 0.0

    scorer = SiriusLikeScorer(
        "CH",
        SiriusLikeScoringConfig(
            db_paired_formulas=frozenset({"C4H8", "C3H6"}),
        ),
    )

    assert scorer.db_paired_score("C3H6", "C4H8") == 1.0
    assert scorer.db_paired_score("C2H4", "C4H8") == 0.0
    assert scorer.db_paired_score("C3H6", "C2H4") == 0.0


def test_sirius_like_db_paired_formula_loader_accepts_text_lists(tmp_path):
    formula_path = tmp_path / "db_formulas.txt"
    formula_path.write_text(
        "# caller-owned formula list\n"
        "C31H36N2O11\n"
        "C22H21NO6,source\n"
        "\n"
    )

    assert load_db_paired_formulas(formula_path) == frozenset(
        {"C31H36N2O11", "C22H21NO6"}
    )


def test_sirius_v6_reference_config_keeps_external_db_map_explicit():
    formulas = frozenset({"C31H36N2O11", "C22H21NO6"})

    config = SiriusLikeScoringConfig.sirius_v6_reference(
        db_paired_formulas=formulas
    )

    assert config.strict_sirius_radical_parity is True
    assert config.enable_common_fragment_score is False
    assert config.enable_carbohydrogen_fragment_score is True
    assert config.use_sirius_tree_size_quality_threshold is True
    assert config.db_paired_formulas is formulas


def test_sirius_v6_reference_disables_learned_common_fragment_table():
    default_scorer = SiriusLikeScorer(
        ["C", "H", "N"],
        SiriusLikeScoringConfig(),
    )
    reference_scorer = SiriusLikeScorer(
        ["C", "H", "N"],
        SiriusLikeScoringConfig.sirius_v6_reference(),
    )

    assert default_scorer.common_fragment_score("C9H7N") > 0.0
    assert reference_scorer.common_fragment_score("C9H7N") == 0.0


def test_sirius_v6_tree_size_quality_uses_processed_peak_count():
    finder = FragmentationTreeFinder("CH")
    root = Fragment(
        formula="C10H20",
        counts=(10, 20),
        ionization="[M+H]+",
        peak_id=0,
        color=0,
        mass=141.0,
        candidate_score=0.0,
    )
    selected = [
        root,
        *(
            Fragment(
                formula=f"C{index}H{2 * index}",
                counts=(index, 2 * index),
                ionization="[M+H]+",
                peak_id=index,
                color=index,
                mass=float(index),
                candidate_score=0.0,
                intensity=1.0,
            )
            for index in range(1, 7)
        ),
    ]
    tree = FragmentationTree(
        root=root,
        fragments=selected,
        losses=[],
        tree_score=0.0,
        is_optimal=True,
        solver_status="Optimal",
        graph_vertex_count=0,
        graph_edge_count=0,
        reduced_vertex_count=0,
        reduced_edge_count=0,
    )
    generated = {
        fragment.formula: _GeneratedCandidate(
            FragmentCandidate(
                fragment.formula,
                fragment.mass,
                peak_id=fragment.peak_id or 0,
                color=fragment.color,
                intensity=1.0,
            ),
            fragment.counts,
            fragment.mass,
        )
        for fragment in selected[1:]
    }
    config = SiriusLikeScoringConfig.sirius_v6_reference()

    assert finder._is_high_quality_tree(
        tree,
        generated,
        config,
        processed_peak_count=7,
    )
    assert not finder._is_high_quality_tree(
        tree,
        generated,
        config,
        processed_peak_count=19,
    )


def test_sirius_like_default_config_uses_padded_search_and_ms2_window():
    config = SiriusLikeScoringConfig()

    assert config.ms2_tolerance_ppm == 10.0
    assert config.mass_deviation_absolute_da == 0.002
    assert config.candidate_search_ppm == 15.0
    assert config.candidate_search_ppm > config.ms2_tolerance_ppm
    assert config.max_fragment_peaks == 59


def test_raw_candidate_generation_uses_absolute_ms2_search_window():
    finder = FragmentationTreeFinder("CHNO")
    config = SiriusLikeScoringConfig()
    scorer = SiriusLikeScorer(finder.element_symbols, config)
    root_counts = finder.parse_formula_counts("C25H47NO9")

    generated = finder._generate_fragment_candidates(
        [
            SpectrumPeak(56.0502, 2933545.0),
            SpectrumPeak(57.0706, 3416372.5),
            SpectrumPeak(60.0450, 6733043.5),
        ],
        root_counts,
        config,
        scorer,
        "C25H47NO9",
        "[M+H]+",
    )

    assert {"C3H5N", "C4H8", "C2H5NO"}.issubset(generated)


def test_raw_candidate_generation_pads_absolute_search_before_sirius_window_filter():
    finder = FragmentationTreeFinder("CHNO")
    config = SiriusLikeScoringConfig.sirius_v6_reference()
    scorer = SiriusLikeScorer(finder.element_symbols, config)
    root_counts = finder.parse_formula_counts("C34H59NO13")

    generated = finder._generate_fragment_candidates(
        [SpectrumPeak(58.0793, 2628.564208984375)],
        root_counts,
        config,
        scorer,
        "C34H59NO13",
        "[M+H]+",
        intensity_scale=19492.771484375,
    )

    assert "C4H9" in generated
    assert abs(generated["C4H9"].candidate.mass - generated["C4H9"].theoretical_mz) <= (
        config.mass_deviation_absolute_da
    )


def test_raw_candidate_generation_allows_sirius_negative_rdbe_fragments():
    finder = FragmentationTreeFinder("CHNO")
    config = SiriusLikeScoringConfig.sirius_v6_reference()
    scorer = SiriusLikeScorer(finder.element_symbols, config)
    root_counts = finder.parse_formula_counts("C34H59N3O9")

    generated = finder._generate_fragment_candidates(
        [SpectrumPeak(214.1411, 113263552.0)],
        root_counts,
        config,
        scorer,
        "C34H59N3O9",
        "[M+H]+",
        intensity_scale=207994960.0,
    )

    assert "C8H21O6" in generated


def test_raw_candidate_scoring_normalizes_intensity_against_parent_peak():
    finder = FragmentationTreeFinder("CHNO")
    config = SiriusLikeScoringConfig()
    scorer = SiriusLikeScorer(finder.element_symbols, config)
    root_counts = finder.parse_formula_counts("C32H36N2O5")
    peaks = [SpectrumPeak(130.0651, 4.0)]

    generated = finder._generate_fragment_candidates(
        peaks,
        root_counts,
        config,
        scorer,
        "C32H36N2O5",
        "[M+H]+",
        intensity_scale=8.0,
    )
    item = next(iter(generated.values()))
    root = _GeneratedCandidate(
        FragmentCandidate("C32H36N2O5", 529.2697, peak_id=0, color=0, intensity=8.0),
        root_counts,
        529.2697,
    )
    scoring = finder._build_sirius_like_scoring(
        root,
        generated,
        scorer,
        intensity_scale=8.0,
    )

    assert item.candidate.intensity == 4.0
    assert scoring.peak_scores[item.candidate.color] == pytest.approx(
        scorer.peak_score(item.candidate.mass, 0.5)
    )


def test_raw_preprocessing_can_merge_close_lower_intensity_peaks():
    finder = FragmentationTreeFinder("CH")
    spectrum = FragmentationSpectrum(
        precursor_mz=150.0,
        precursor_formula="C10H22",
        peaks=[
            SpectrumPeak(100.0000, 10.0),
            SpectrumPeak(100.0010, 20.0),
            SpectrumPeak(110.0, 5.0),
        ],
    )

    _, merged_fragments = finder._split_root_peak(spectrum, SiriusLikeScoringConfig())
    _, unmerged_fragments = finder._split_root_peak(
        spectrum,
        SiriusLikeScoringConfig(merge_close_peaks=False),
    )

    assert [peak.mz for peak in merged_fragments] == [100.0010, 110.0]
    assert [peak.mz for peak in unmerged_fragments] == [100.0000, 100.0010, 110.0]


def test_raw_preprocessing_default_peak_limit_is_parent_inclusive():
    finder = FragmentationTreeFinder("CH")
    spectrum = FragmentationSpectrum(
        precursor_mz=500.0,
        precursor_formula="C40H80",
        peaks=[
            *(SpectrumPeak(100.0 + index, float(index + 1)) for index in range(60)),
            SpectrumPeak(500.0, 100.0),
        ],
    )

    root_peak, fragments = finder._split_root_peak(spectrum, SiriusLikeScoringConfig())

    assert root_peak is not None
    assert len(fragments) == 58
    assert {100.0, 101.0} not in {peak.mz for peak in fragments}


def test_raw_preprocessing_keeps_weak_parent_outside_peak_limit():
    finder = FragmentationTreeFinder("CH")
    spectrum = FragmentationSpectrum(
        precursor_mz=500.0,
        precursor_formula="C40H80",
        peaks=[
            *(SpectrumPeak(100.0 + index, float(index + 1)) for index in range(60)),
            SpectrumPeak(500.0, 0.5),
        ],
    )

    root_peak, fragments = finder._split_root_peak(spectrum, SiriusLikeScoringConfig())

    assert root_peak is not None
    assert len(fragments) == 59
    assert 100.0 not in {peak.mz for peak in fragments}


def test_raw_candidate_generation_disjoins_nearby_duplicate_decompositions():
    finder = FragmentationTreeFinder("CH")
    peaks = [SpectrumPeak(100.0, 1.0), SpectrumPeak(100.001, 1.0)]
    candidates_by_peak = [
        {
            "C2H4": _GeneratedCandidate(
                FragmentCandidate("C2H4", 100.0, peak_id=1, color=1),
                (2, 4),
                100.0004,
            )
        },
        {
            "C2H4": _GeneratedCandidate(
                FragmentCandidate("C2H4", 100.001, peak_id=2, color=2),
                (2, 4),
                100.0008,
            )
        },
    ]

    finder._disjoin_nearby_fragment_candidates(
        peaks,
        candidates_by_peak,
        SiriusLikeScoringConfig(),
    )

    assert "C2H4" not in candidates_by_peak[0]
    assert "C2H4" in candidates_by_peak[1]


def test_sirius_like_carbohydrogen_scores_match_v6_default_plugin():
    default_scorer = SiriusLikeScorer("CHNO", SiriusLikeScoringConfig())
    scorer = SiriusLikeScorer(
        "CHNO",
        SiriusLikeScoringConfig(enable_carbohydrogen_fragment_score=True),
    )
    cho_counts = (8, 12, 0, 1)
    chno_counts = (8, 12, 1, 1)

    assert scorer.carbohydrogen_root_score(cho_counts) == 2.5
    assert scorer.carbohydrogen_root_score(chno_counts) == 0.0
    assert default_scorer.carbohydrogen_fragment_score(cho_counts, 0.5) == 0.0
    assert scorer.carbohydrogen_fragment_score(cho_counts, 0.02) == 0.0
    assert scorer.carbohydrogen_fragment_score(chno_counts, 0.5) == 0.0
    assert scorer.carbohydrogen_fragment_score(cho_counts, 0.5) == pytest.approx(0.5)


def test_sirius_like_common_loss_scores_match_v6_recombination_edge_cases():
    scorer = SiriusLikeScorer("CHNO", SiriusLikeScoringConfig())

    assert scorer.common_loss_score("C2H6") == pytest.approx(-1.3646611753318298)
    assert scorer.common_loss_score("C2H4O") == pytest.approx(-0.6057228211483281)
    assert scorer.common_loss_score("C15H21NO7") == pytest.approx(2.4976227503252453)
    assert scorer.common_loss_score("C3H2O") == pytest.approx(0.19848183290332289)


def test_sirius_like_multimere_and_fatty_acid_chain_loss_scores_match_v6():
    scorer = SiriusLikeScorer("CHNO", SiriusLikeScoringConfig())
    c7h8o_counts = (7, 8, 0, 1)

    assert scorer.multimere_loss_score("C7H8O", "C7H8O", False) == 2.0
    assert scorer.multimere_loss_score("C7H8O", "C7H8O", True) == 10.0
    assert scorer.multimere_loss_score("C7H8O", "C8H8O", False) == 0.0
    assert scorer.lipid_chain_from_counts(c7h8o_counts) == (7, 2)
    assert scorer.fatty_acid_chain_loss_score("C7H8O", c7h8o_counts) == pytest.approx(
        0.1657309333723955
    )


def test_sirius_like_free_radical_scoring_matches_v6_negative_rdbe_parity():
    default_scorer = SiriusLikeScorer("CHNO", SiriusLikeScoringConfig())
    scorer = SiriusLikeScorer(
        "CHNO",
        SiriusLikeScoringConfig(strict_sirius_radical_parity=True),
    )

    assert scorer.doubled_rdbe((1, 6, 1, 0)) == -1
    assert default_scorer.free_radical_loss_score("CH6N", (1, 6, 1, 0)) == pytest.approx(
        -2.2909585508352253
    )
    assert scorer.free_radical_loss_score("CH6N", (1, 6, 1, 0)) == pytest.approx(
        0.011626542158820332
    )
    assert scorer.free_radical_loss_score("CH3", (1, 3, 0, 0)) == pytest.approx(
        -0.09373397349900595
    )
    assert scorer.free_radical_loss_score("CH2N", (1, 2, 1, 0)) == pytest.approx(
        -2.2909585508352253
    )


def test_rust_fragmentation_tree_python_bridge_solves_colorful_tree():
    root = [("C4H8", [4, 8], "[M+H]+", 0, 0, 56.0, 1.0)]
    fragments = [
        ("C3H6", [3, 6], "[M+H]+", 1, 1, 42.0, 3.0),
        ("C2H4", [2, 4], "[M+H]+", 2, 2, 28.0, 5.0),
        ("CH2", [1, 2], "[M+H]+", 3, 3, 14.0, -30.0),
    ]

    (
        tree_weight,
        is_optimal,
        status,
        root_formula,
        selected_formulas,
        selected_losses,
        graph_vertices,
        graph_edges,
        reduced_vertices,
        reduced_edges,
    ) = rust.solve_fragmentation_tree_python(
        root,
        fragments,
        peak_scores=[(1, 4.0), (2, 7.0)],
        peak_pair_scores=[(0, 1, 1.0), (1, 2, 2.0)],
    )

    assert tree_weight == 23.0
    assert is_optimal
    assert status == "Optimal"
    assert root_formula == "C4H8"
    assert selected_formulas == ["C4H8", "C3H6", "C2H4"]
    assert selected_losses == [
        ("", "C4H8", 1.0),
        ("C4H8", "C3H6", 8.0),
        ("C3H6", "C2H4", 14.0),
    ]
    assert (graph_vertices, graph_edges) == (5, 7)
    assert (reduced_vertices, reduced_edges) == (4, 4)


def test_public_fragmentation_tree_api_solves_colorful_tree():
    finder = FragmentationTreeFinder("CH")
    root = [
        FragmentCandidate(
            formula="C4H8",
            mass=56.0,
            score=1.0,
            peak_id=0,
            color=0,
        )
    ]
    fragments = [
        FragmentCandidate("C3H6", 42.0, score=3.0, peak_id=1, color=1),
        FragmentCandidate("C2H4", 28.0, score=5.0, peak_id=2, color=2),
        FragmentCandidate("CH2", 14.0, score=-30.0, peak_id=3, color=3),
    ]

    tree = finder.find_tree(
        root,
        fragments,
        scoring=ExplicitFragmentationScoring(
            peak_scores={1: 4.0, 2: 7.0},
            peak_pair_scores={(0, 1): 1.0, (1, 2): 2.0},
        ),
    )

    assert tree.tree_score == 23.0
    assert tree.is_optimal
    assert tree.solver_status == "Optimal"
    assert tree.root.formula == "C4H8"
    assert [fragment.formula for fragment in tree.fragments] == [
        "C4H8",
        "C3H6",
        "C2H4",
    ]
    assert [
        (loss.source.formula, loss.target.formula, loss.formula, loss.score)
        for loss in tree.losses
    ] == [
        ("C4H8", "C3H6", "CH2", 8.0),
        ("C3H6", "C2H4", "CH2", 14.0),
    ]
    assert (tree.graph_vertex_count, tree.graph_edge_count) == (5, 7)
    assert (tree.reduced_vertex_count, tree.reduced_edge_count) == (4, 4)
    assert "C4H8" in tree.to_table()
    assert "CH2" in tree.losses_table()


NOVOBIOCIN_SYMBOLS = ["C", "H", "N", "O"]

NOVOBIOCIN_SIRIUS_FRAGMENTS = [
    ("C31H36N2O11", 613.2512, -1.9934733245291434, 0),
    ("C25H20N2O3", 397.1508, 3.410887797531137, 1),
    ("C25H19N2O3", 396.148, 5.678059274210523, 2),
    ("C12H13N2O2", 218.1051, 5.907809282497401, 3),
    ("C12H11N2O", 200.0943, 3.7163099700313027, 4),
    ("C12H12O2", 189.0925, 14.367420192334787, 5),
    ("C11H9N2O", 186.0781, 4.5523190844709, 6),
    ("C11H10N", 157.087, 4.268712393324276, 7),
    ("C10H8N", 143.0719, 3.966987683601832, 8),
    ("C7H8O2", 125.0601, 7.867544658102682, 9),
]

NOVOBIOCIN_SIRIUS_LOSSES = {
    ("C31H36N2O11", "C25H20N2O3"): -8.213343168428171,
    ("C25H20N2O3", "C25H19N2O3"): -2.84780358455102,
    ("C25H19N2O3", "C12H13N2O2"): -5.043497502143028,
    ("C12H13N2O2", "C12H11N2O"): 1.5224573668487713,
    ("C12H13N2O2", "C12H12O2"): -4.0665408684181745,
    ("C12H13N2O2", "C11H9N2O"): -1.2056289684540389,
    ("C12H11N2O", "C11H10N"): -1.5806418664147743,
    ("C11H9N2O", "C10H8N"): -1.5806376675416292,
    ("C12H12O2", "C7H8O2"): -1.8994978850669497,
}

NOVOBIOCIN_SIRIUS_V6_RAW_REFERENCE = {
    "tree_score": 33.068402054104155,
    "formulas": {
        "C31H36N2O11",
        "C22H21NO6",
        "C12H13N2O2",
        "C12H11N2O",
        "C12H12O2",
        "C11H9N2O",
        "C11H10N",
        "C10H8N",
        "C7H8O2",
    },
    "losses": {
        ("C31H36N2O11", "C22H21NO6", "C9H15NO5"),
        ("C31H36N2O11", "C12H13N2O2", "C19H23O9"),
        ("C12H13N2O2", "C12H11N2O", "H2O"),
        ("C22H21NO6", "C12H12O2", "C10H9NO4"),
        ("C12H13N2O2", "C11H9N2O", "CH4O"),
        ("C12H11N2O", "C11H10N", "CHNO"),
        ("C12H11N2O", "C10H8N", "C2H3NO"),
        ("C12H12O2", "C7H8O2", "C5H4"),
    },
}

MASSBANK_FIXTURE = Path(__file__).parent / "data" / "fragmentation_massbank_records.json"


def _novobiocin_sirius_inputs():
    counts = {
        formula: list(map(int, rust.parse_formula_counts(formula, NOVOBIOCIN_SYMBOLS)))
        for formula, *_ in NOVOBIOCIN_SIRIUS_FRAGMENTS
    }

    def is_subformula(parent: str, child: str) -> bool:
        return all(
            parent_count >= child_count
            for parent_count, child_count in zip(counts[parent], counts[child])
        )

    root_formula, root_mass, root_score, root_color = NOVOBIOCIN_SIRIUS_FRAGMENTS[0]
    root = [
        (
            root_formula,
            counts[root_formula],
            "[M+H]+",
            root_color,
            root_color,
            root_mass,
            root_score,
        )
    ]
    fragments = [
        (formula, counts[formula], "[M+H]+", color, color, mass, score)
        for formula, mass, score, color in NOVOBIOCIN_SIRIUS_FRAGMENTS[1:]
    ]

    loss_scores = []
    for parent, child in itertools.product(
        [row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS],
        [row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS[1:]],
    ):
        if parent != child and is_subformula(parent, child):
            loss_scores.append(
                (
                    parent,
                    child,
                    NOVOBIOCIN_SIRIUS_LOSSES.get((parent, child), -1_000_000.0),
                )
            )

    return root, fragments, loss_scores


def _novobiocin_public_inputs():
    finder = FragmentationTreeFinder(NOVOBIOCIN_SYMBOLS)

    root_formula, root_mass, root_score, root_color = NOVOBIOCIN_SIRIUS_FRAGMENTS[0]
    root = [
        FragmentCandidate(
            root_formula,
            root_mass,
            score=root_score,
            peak_id=root_color,
            color=root_color,
        )
    ]
    fragments = [
        FragmentCandidate(
            formula,
            mass,
            score=score,
            peak_id=color,
            color=color,
        )
        for formula, mass, score, color in NOVOBIOCIN_SIRIUS_FRAGMENTS[1:]
    ]

    loss_scores = {}
    formulas = [row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS]
    for parent, child in itertools.product(formulas, formulas[1:]):
        if parent != child and finder.is_subformula(parent, child):
            loss_scores[(parent, child)] = NOVOBIOCIN_SIRIUS_LOSSES.get(
                (parent, child),
                -1_000_000.0,
            )

    return finder, root, fragments, ExplicitFragmentationScoring(loss_scores=loss_scores)


def test_novobiocin_sirius_scored_tree_matches_reference():
    root, fragments, loss_scores = _novobiocin_sirius_inputs()

    (
        tree_weight,
        is_optimal,
        status,
        root_formula,
        selected_formulas,
        selected_losses,
        graph_vertices,
        graph_edges,
        reduced_vertices,
        reduced_edges,
    ) = rust.solve_fragmentation_tree_python(
        root,
        fragments,
        loss_scores=loss_scores,
    )

    assert tree_weight == pytest.approx(26.827442867406685)
    assert is_optimal
    assert status == "Optimal"
    assert root_formula == "C31H36N2O11"
    assert set(selected_formulas) == {row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS}
    assert [(source, target) for source, target, _ in selected_losses] == [
        ("", "C31H36N2O11"),
        ("C31H36N2O11", "C25H20N2O3"),
        ("C25H20N2O3", "C25H19N2O3"),
        ("C25H19N2O3", "C12H13N2O2"),
        ("C12H13N2O2", "C12H11N2O"),
        ("C12H13N2O2", "C12H12O2"),
        ("C12H13N2O2", "C11H9N2O"),
        ("C12H11N2O", "C11H10N"),
        ("C11H9N2O", "C10H8N"),
        ("C12H12O2", "C7H8O2"),
    ]
    assert (graph_vertices, graph_edges) == (11, 37)
    assert (reduced_vertices, reduced_edges) == (11, 10)


def test_public_api_novobiocin_sirius_scored_tree_matches_reference():
    finder, root, fragments, scoring = _novobiocin_public_inputs()

    tree = finder.find_tree(root, fragments, scoring=scoring)

    assert tree.tree_score == pytest.approx(26.827442867406685)
    assert tree.is_optimal
    assert tree.solver_status == "Optimal"
    assert tree.root.formula == "C31H36N2O11"
    assert {fragment.formula for fragment in tree.fragments} == {
        row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS
    }
    assert [
        (loss.source.formula, loss.target.formula, loss.formula)
        for loss in tree.losses
    ] == [
        ("C31H36N2O11", "C25H20N2O3", "C6H16O8"),
        ("C25H20N2O3", "C25H19N2O3", "H"),
        ("C25H19N2O3", "C12H13N2O2", "C13H6O"),
        ("C12H13N2O2", "C12H11N2O", "H2O"),
        ("C12H13N2O2", "C12H12O2", "HN2"),
        ("C12H13N2O2", "C11H9N2O", "CH4O"),
        ("C12H11N2O", "C11H10N", "CHNO"),
        ("C11H9N2O", "C10H8N", "CHNO"),
        ("C12H12O2", "C7H8O2", "C5H4"),
    ]
    assert (tree.graph_vertex_count, tree.graph_edge_count) == (11, 37)
    assert (tree.reduced_vertex_count, tree.reduced_edge_count) == (11, 10)
    assert "C31H36N2O11" in tree.to_table()
    assert "C6H16O8" in tree.losses_table()


SIRIUS_RAW_REFERENCE_TARGETS = {
    "MSBNK-Athens_Univ-AU276702": {
        "name": "acetaminophen",
        "tree_score": 10.777285849243086,
        "formulas": {
            "C8H9NO2",
            "C8H7NO",
            "C6H7NO",
            "C6H6NO",
        },
        "losses": {
            ("C8H9NO2", "C8H7NO", "H2O"),
            ("C8H9NO2", "C6H7NO", "C2H2O"),
            ("C8H9NO2", "C6H6NO", "C2H3O"),
        },
    },
    "MSBNK-ACES_SU-AS000913": {
        "name": "caffeine",
        "tree_score": 46.759362437290264,
        "formulas": {
            "C8H10N4O2",
            "C7H7N4O2",
            "C5H6N4O",
            "C6H7N3O",
            "C5H4N3O",
            "C5H6N2O",
            "C5H7N3",
            "C5H4N2O",
            "C4H6N2",
            "C3H4N2",
            "C3H5N",
        },
        "losses": {
            ("C8H10N4O2", "C7H7N4O2", "CH3"),
            ("C8H10N4O2", "C5H6N4O", "C3H4O"),
            ("C8H10N4O2", "C6H7N3O", "C2H3NO"),
            ("C6H7N3O", "C5H4N3O", "CH3"),
            ("C6H7N3O", "C5H6N2O", "CHN"),
            ("C6H7N3O", "C5H7N3", "CO"),
            ("C5H6N2O", "C5H4N2O", "H2"),
            ("C5H6N2O", "C4H6N2", "CO"),
            ("C5H6N2O", "C3H4N2", "C2H2O"),
            ("C4H6N2", "C3H5N", "CHN"),
        },
    },
    "MSBNK-Athens_Univ-AU110802": {
        "name": "propranolol",
        "tree_score": 74.76887380398153,
        "formulas": {
            "C16H21NO2",
            "C16H19NO",
            "C13H15NO2",
            "C13H10O",
            "C13H8",
            "C11H8O",
            "C12H10",
            "C12H8",
            "C10H8O",
            "C11H8",
            "C10H8",
            "C6H13NO",
        },
        "losses": {
            ("C16H21NO2", "C16H19NO", "H2O"),
            ("C16H21NO2", "C13H15NO2", "C3H6"),
            ("C16H19NO", "C13H10O", "C3H9N"),
            ("C13H10O", "C13H8", "H2O"),
            ("C13H10O", "C11H8O", "C2H2"),
            ("C13H10O", "C12H10", "CO"),
            ("C12H10", "C12H8", "H2"),
            ("C13H15NO2", "C10H8O", "C3H7NO"),
            ("C13H10O", "C11H8", "C2H2O"),
            ("C11H8O", "C10H8", "CO"),
            ("C13H15NO2", "C6H13NO", "C7H2O"),
        },
    },
    "MSBNK-Athens_Univ-AU121202": {
        "name": "lidocaine",
        "tree_score": 11.320980621477784,
        "formulas": {
            "C14H22N2O",
            "C14H20N2",
            "C10H11NO",
            "C9H11N",
        },
        "losses": {
            ("C14H22N2O", "C14H20N2", "H2O"),
            ("C14H22N2O", "C10H11NO", "C4H11N"),
            ("C10H11NO", "C9H11N", "CO"),
        },
    },
}


def test_raw_massbank_novobiocin_matches_sirius_v6_tree():
    records = {
        record["accession"]: record
        for record in json.loads(MASSBANK_FIXTURE.read_text())
    }
    spectrum = FragmentationSpectrum.from_massbank_record(
        records["MSBNK-Athens_Univ-AU116706"]
    )
    finder = FragmentationTreeFinder("CHNO")

    tree = finder.find_tree_from_spectrum(spectrum)
    selected = {fragment.formula for fragment in tree.fragments}
    selected_losses = {
        (loss.source.formula, loss.target.formula, loss.formula)
        for loss in tree.losses
    }
    reference = NOVOBIOCIN_SIRIUS_V6_RAW_REFERENCE["formulas"]
    reference_losses = NOVOBIOCIN_SIRIUS_V6_RAW_REFERENCE["losses"]

    assert tree.root.formula == "C31H36N2O11"
    assert tree.is_optimal
    assert selected == reference
    assert selected_losses == reference_losses


def test_raw_massbank_novobiocin_strict_parity_matches_with_external_db_map():
    records = {
        record["accession"]: record
        for record in json.loads(MASSBANK_FIXTURE.read_text())
    }
    spectrum = FragmentationSpectrum.from_massbank_record(
        records["MSBNK-Athens_Univ-AU116706"]
    )
    finder = FragmentationTreeFinder("CHNO")

    tree = finder.find_tree_from_spectrum(
        spectrum,
        scoring_config=SiriusLikeScoringConfig(
            strict_sirius_radical_parity=True,
            db_paired_formulas=frozenset({"C31H36N2O11", "C22H21NO6"}),
        ),
    )
    selected = {fragment.formula for fragment in tree.fragments}
    selected_losses = {
        (loss.source.formula, loss.target.formula, loss.formula)
        for loss in tree.losses
    }

    assert selected == NOVOBIOCIN_SIRIUS_V6_RAW_REFERENCE["formulas"]
    assert selected_losses == NOVOBIOCIN_SIRIUS_V6_RAW_REFERENCE["losses"]


@pytest.mark.parametrize(
    ("accession", "reference"),
    list(SIRIUS_RAW_REFERENCE_TARGETS.items()),
    ids=[target["name"] for target in SIRIUS_RAW_REFERENCE_TARGETS.values()],
)
def test_raw_massbank_reference_targets_match_sirius_v6_trees(
    accession,
    reference,
):
    records = {
        record["accession"]: record
        for record in json.loads(MASSBANK_FIXTURE.read_text())
    }
    record = records[accession]
    spectrum = FragmentationSpectrum.from_massbank_record(record)
    finder = FragmentationTreeFinder("CHNO")

    tree = finder.find_tree_from_spectrum(spectrum)
    selected = {fragment.formula for fragment in tree.fragments}
    selected_losses = {
        (loss.source.formula, loss.target.formula, loss.formula)
        for loss in tree.losses
    }

    assert tree.root.formula == spectrum.precursor_formula
    assert tree.is_optimal
    assert len(tree.losses) == len(tree.fragments) - 1
    assert selected == reference["formulas"]
    assert selected_losses == reference["losses"]
