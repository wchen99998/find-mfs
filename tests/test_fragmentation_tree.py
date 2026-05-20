from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from find_mfs.fragmentation import (
    ExplicitFragmentationScoring,
    FragmentCandidate,
    FragmentationSpectrum,
    FragmentationTreeFinder,
)

rust = pytest.importorskip("find_mfs._rust")


def test_public_fragmentation_api_is_exported_from_package():
    from find_mfs import (
        ExplicitFragmentationScoring as TopLevelScoring,
        FragmentCandidate as TopLevelFragmentCandidate,
        FragmentationTreeFinder as TopLevelFragmentationTreeFinder,
    )

    assert TopLevelFragmentCandidate is FragmentCandidate
    assert TopLevelFragmentationTreeFinder is FragmentationTreeFinder
    assert TopLevelScoring is ExplicitFragmentationScoring


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
        "tree_score": 6.164391034221983,
        "formulas": {
            "C8H9NO2",
            "C8H7NO",
            "C6H7NO",
        },
        "losses": {
            ("C8H9NO2", "C8H7NO", "H2O"),
            ("C8H9NO2", "C6H7NO", "C2H2O"),
        },
        "min_formula_overlap": 3,
        "min_loss_overlap": 2,
    },
    "MSBNK-ACES_SU-AS000913": {
        "name": "caffeine",
        "tree_score": 36.631475583426344,
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
        "min_formula_overlap": 10,
        "min_loss_overlap": 9,
    },
    "MSBNK-Athens_Univ-AU110802": {
        "name": "propranolol",
        "tree_score": 40.56118139783991,
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
        "min_formula_overlap": 11,
        "min_loss_overlap": 10,
    },
    "MSBNK-Athens_Univ-AU121202": {
        "name": "lidocaine",
        "tree_score": 3.2725929705549333,
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
        "min_formula_overlap": 4,
        "min_loss_overlap": 3,
    },
}


def test_raw_massbank_novobiocin_reconstructs_rough_sirius_tree():
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
    reference = {row[0] for row in NOVOBIOCIN_SIRIUS_FRAGMENTS}

    assert tree.root.formula == "C31H36N2O11"
    assert tree.is_optimal
    assert len(selected & reference) >= 7
    assert {
        "C12H13N2O2",
        "C12H11N2O",
        "C12H12O2",
        "C11H9N2O",
        "C10H8N",
        "C7H8O2",
    }.issubset(selected)


@pytest.mark.parametrize(
    ("accession", "reference"),
    list(SIRIUS_RAW_REFERENCE_TARGETS.items()),
    ids=[target["name"] for target in SIRIUS_RAW_REFERENCE_TARGETS.values()],
)
def test_raw_massbank_reference_targets_roughly_match_sirius_trees(
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
    assert len(selected & reference["formulas"]) >= reference["min_formula_overlap"]
    assert len(selected_losses & reference["losses"]) >= reference["min_loss_overlap"]
