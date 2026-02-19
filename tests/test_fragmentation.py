"""
Tests for fragmentation tree construction.

Includes tests with erythromycin — a well-characterized macrolide antibiotic
whose CID fragmentation is extensively documented in the literature, making
it a good benchmark for fragmentation tree algorithms.
"""
import numpy as np
import pytest
from molmass import Formula

from find_mfs.core.finder import FormulaFinder
from find_mfs.fragmentation.tree import (
    FragmentationTreeGenerator,
    FragmentationTree,
    FragmentNode,
    FragmentEdge,
    Peak,
    IsotopeEnvelope,
    detect_isotope_envelopes,
    _is_subformula,
    _formula_to_count_vector,
    _counts_to_formula_string,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chnops_finder():
    return FormulaFinder("CHNOPS")


@pytest.fixture(scope="module")
def generator(chnops_finder):
    return FragmentationTreeGenerator(
        chnops_finder,
        error_ppm=10.0,
        max_candidates_per_peak=5,
    )


# ---------------------------------------------------------------------------
# IsotopeEnvelope tests
# ---------------------------------------------------------------------------

class TestIsotopeEnvelope:

    def test_basic_construction(self):
        peaks = np.array([[100.0, 1000.0], [101.003, 110.0]])
        env = IsotopeEnvelope(peaks=peaks)
        assert env.monoisotopic_mz == pytest.approx(100.0)
        assert env.monoisotopic_intensity == pytest.approx(1000.0)
        assert env.peaks.shape == (2, 2)

    def test_explicit_monoisotopic(self):
        peaks = np.array([[100.0, 800.0], [101.003, 110.0]])
        env = IsotopeEnvelope(
            peaks=peaks,
            monoisotopic_mz=100.0,
            monoisotopic_intensity=800.0,
        )
        assert env.monoisotopic_mz == pytest.approx(100.0)
        assert env.monoisotopic_intensity == pytest.approx(800.0)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            IsotopeEnvelope(peaks=np.array([1.0, 2.0, 3.0]))

    def test_single_peak_envelope(self):
        env = IsotopeEnvelope(peaks=np.array([[200.0, 500.0]]))
        assert env.peaks.shape == (1, 2)
        assert env.monoisotopic_mz == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# detect_isotope_envelopes tests
# ---------------------------------------------------------------------------

class TestDetectIsotopeEnvelopes:

    def test_simple_pair(self):
        """Two peaks ~1.003 Da apart should be grouped."""
        spectrum = [(100.0, 1000.0), (101.003, 100.0)]
        envs = detect_isotope_envelopes(spectrum, charge=1)
        assert len(envs) == 1
        assert envs[0].peaks.shape[0] == 2

    def test_no_grouping_for_distant_peaks(self):
        """Peaks far apart should not be grouped."""
        spectrum = [(100.0, 1000.0), (200.0, 800.0)]
        envs = detect_isotope_envelopes(spectrum, charge=1)
        assert len(envs) == 2
        assert all(env.peaks.shape[0] == 1 for env in envs)

    def test_h2_loss_not_grouped(self):
        """
        A peak at ~M+2 with high relative intensity should NOT be grouped
        as an isotopologue — it's more likely an H2 loss.
        """
        # M+0 at 100.0, M+1 at 101.003 (normal isotopologue),
        # M+2 candidate at 102.006 but with 80% relative intensity
        # (way too high for a natural M+2 isotopologue of a small molecule)
        spectrum = [
            (100.0, 1000.0),
            (101.003, 100.0),
            (102.006, 800.0),  # too intense to be M+2
        ]
        envs = detect_isotope_envelopes(
            spectrum, charge=1, h2_loss_intensity_ratio=0.5
        )
        # The 102.006 peak should be its own envelope
        assert any(
            env.peaks.shape[0] == 2
            and env.monoisotopic_mz == pytest.approx(100.0, abs=0.01)
            for env in envs
        )
        assert any(
            env.monoisotopic_mz == pytest.approx(102.006, abs=0.01)
            for env in envs
        )

    def test_doubly_charged_spacing(self):
        """For z=2, isotope spacing should be ~0.5 Da."""
        spectrum = [(100.0, 1000.0), (100.502, 100.0)]
        envs = detect_isotope_envelopes(spectrum, charge=2)
        assert len(envs) == 1
        assert envs[0].peaks.shape[0] == 2

    def test_empty_spectrum(self):
        envs = detect_isotope_envelopes([], charge=1)
        assert len(envs) == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    def test_is_subformula(self):
        parent = np.array([6, 12, 0, 6, 0, 0], dtype=np.int64)
        child = np.array([3, 6, 0, 3, 0, 0], dtype=np.int64)
        assert _is_subformula(child, parent)
        assert not _is_subformula(parent, child)

    def test_counts_to_formula_string(self):
        symbols = ["C", "H", "N", "O", "P", "S"]
        counts = np.array([6, 12, 0, 6, 0, 0], dtype=np.int64)
        result = _counts_to_formula_string(symbols, counts)
        assert result == "C6H12O6"

    def test_counts_to_formula_string_single_count(self):
        symbols = ["C", "H", "N", "O", "P", "S"]
        counts = np.array([1, 2, 0, 1, 0, 0], dtype=np.int64)
        result = _counts_to_formula_string(symbols, counts)
        assert result == "CH2O"


# ---------------------------------------------------------------------------
# FragmentationTreeGenerator basic tests
# ---------------------------------------------------------------------------

class TestFragmentationTreeGenerator:

    def test_initialization(self, chnops_finder):
        gen = FragmentationTreeGenerator(chnops_finder)
        assert gen.error_ppm == 10.0
        assert gen.max_peaks == 40
        assert "H2O" in gen.common_losses

    def test_custom_common_losses(self, chnops_finder):
        gen = FragmentationTreeGenerator(
            chnops_finder,
            common_losses=["H2O", "CO2"],
        )
        assert gen.common_losses == ("H2O", "CO2")

    def test_generate_requires_input(self, generator):
        with pytest.raises(ValueError, match="Either"):
            generator.generate(precursor_mz=100.0)

    def test_generate_rejects_both_inputs(self, generator):
        with pytest.raises(ValueError, match="not both"):
            generator.generate(
                precursor_mz=100.0,
                spectrum=[(50.0, 100.0)],
                envelopes=[
                    IsotopeEnvelope(peaks=np.array([[50.0, 100.0]]))
                ],
            )

    def test_simple_tree_from_spectrum(self, generator):
        """Build a tree for a simple case: glucose [M+H]+."""
        glucose_mh = Formula("C6H12O6H+").monoisotopic_mass

        # Synthetic MS/MS spectrum for glucose [M+H]+
        # Common fragments: loss of H2O (×1,2,3), loss of CH2O, etc.
        spectrum = [
            (glucose_mh - 18.010, 800.0),    # -H2O
            (glucose_mh - 36.021, 500.0),    # -2×H2O
            (glucose_mh - 54.032, 200.0),    # -3×H2O
            (glucose_mh - 30.011, 300.0),    # -CH2O
        ]

        tree = generator.generate(
            precursor_mz=glucose_mh,
            spectrum=spectrum,
            charge=1,
            adduct="H",
            auto_detect_envelopes=False,
        )

        assert isinstance(tree, FragmentationTree)
        assert tree.total_score > -float("inf")
        assert len(tree.nodes) >= 1  # at least root

    def test_simple_tree_from_envelopes(self, generator):
        """Build a tree using pre-assembled isotope envelopes."""
        glucose_mh = Formula("C6H12O6H+").monoisotopic_mass

        # Envelopes with M+0 and M+1 for each fragment
        envs = [
            IsotopeEnvelope(peaks=np.array([
                [glucose_mh - 18.010, 800.0],
                [glucose_mh - 18.010 + 1.003, 55.0],
            ])),
            IsotopeEnvelope(peaks=np.array([
                [glucose_mh - 36.021, 500.0],
                [glucose_mh - 36.021 + 1.003, 30.0],
            ])),
        ]

        tree = generator.generate(
            precursor_mz=glucose_mh,
            envelopes=envs,
            charge=1,
            adduct="H",
        )

        assert isinstance(tree, FragmentationTree)
        assert len(tree.nodes) >= 1

    def test_tree_table_output(self, generator):
        """Test the to_table() method produces readable output."""
        glucose_mh = Formula("C6H12O6H+").monoisotopic_mass
        spectrum = [
            (glucose_mh - 18.010, 800.0),
            (glucose_mh - 36.021, 500.0),
        ]

        tree = generator.generate(
            precursor_mz=glucose_mh,
            spectrum=spectrum,
            charge=1,
            adduct="H",
            auto_detect_envelopes=False,
        )

        table = tree.to_table()
        assert isinstance(table, str)
        assert "FragmentationTree" in table

    def test_tree_respects_subformula_constraint(self, generator):
        """All fragment nodes must be subformulae of the root."""
        glucose_mh = Formula("C6H12O6H+").monoisotopic_mass
        spectrum = [
            (glucose_mh - 18.010, 800.0),
            (glucose_mh - 30.011, 300.0),
        ]

        tree = generator.generate(
            precursor_mz=glucose_mh,
            spectrum=spectrum,
            charge=1,
            adduct="H",
            auto_detect_envelopes=False,
        )

        root = [n for n in tree.nodes if n.idx == tree.root_idx][0]
        for node in tree.nodes:
            if node.idx == tree.root_idx:
                continue
            assert _is_subformula(node.counts, root.counts), (
                f"Node {node.formula.formula} is not a subformula of "
                f"root {root.formula.formula}"
            )


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:

    def test_common_loss_bonus(self, chnops_finder):
        gen = FragmentationTreeGenerator(
            chnops_finder,
            loss_common_bonus=2.0,
            loss_mass_deviation_scale=0.0,
        )
        score_h2o = gen._score_edge(
            loss_formula="H2O", loss_mass=18.010, observed_delta_mz=18.010
        )
        score_random = gen._score_edge(
            loss_formula="C3H5NO2", loss_mass=87.032, observed_delta_mz=87.032
        )
        # H2O should get the common-loss bonus
        assert score_h2o > score_random

    def test_mass_deviation_penalty(self, chnops_finder):
        gen = FragmentationTreeGenerator(
            chnops_finder,
            error_ppm=10.0,
            loss_mass_deviation_scale=1.0,
        )
        # Perfect mass match
        score_good = gen._score_edge(
            loss_formula="H2O",
            loss_mass=18.010565,
            observed_delta_mz=18.010565,
        )
        # Poor mass match (0.1 Da off)
        score_bad = gen._score_edge(
            loss_formula="H2O",
            loss_mass=18.010565,
            observed_delta_mz=18.110565,
        )
        assert score_good > score_bad

    def test_large_loss_penalized(self, chnops_finder):
        gen = FragmentationTreeGenerator(
            chnops_finder,
            loss_mass_scale=50.0,
            loss_mass_deviation_scale=0.0,
        )
        # Small loss
        score_small = gen._score_edge(
            loss_formula="H2O", loss_mass=18.0, observed_delta_mz=18.0
        )
        # Large loss (not common)
        score_large = gen._score_edge(
            loss_formula="C10H20O5", loss_mass=220.0, observed_delta_mz=220.0
        )
        assert score_small > score_large


# ---------------------------------------------------------------------------
# Erythromycin fragmentation test
# ---------------------------------------------------------------------------

class TestErythromycin:
    """
    Test fragmentation tree construction using erythromycin A — a macrolide
    antibiotic (C37H67NO13, MW 733.461) whose CID fragmentation is very
    well characterized in the literature.

    Key [M+H]+ fragments (positive mode, z=1):
      - m/z 576.375: loss of cladinose sugar (-C8H14O3, ~158.094)
      - m/z 558.365: loss of cladinose + H2O
      - m/z 158.118: protonated desosamine (C8H16NO2)+

    Reference: Holzgrabe & Nap (2003), J. Pharm. Biomed. Anal.

    Note: For large molecules (>500 Da) like erythromycin, max_counts
    constraints are essential to keep the candidate space manageable.
    Without constraints, CHNOPS decomposition at 734 Da yields thousands
    of candidates and the correct formula gets buried.
    """

    # Erythromycin A: C37H67NO13
    PRECURSOR_MZ = Formula("C37H68NO13+").monoisotopic_mass  # [M+H]+
    MAX_COUNTS = {"C": 40, "H": 70, "N": 2, "O": 15, "P": 0, "S": 0}

    @pytest.fixture(scope="class")
    def erythromycin_finder(self):
        return FormulaFinder("CHNOPS")

    @pytest.fixture(scope="class")
    def erythromycin_generator(self, erythromycin_finder):
        return FragmentationTreeGenerator(
            erythromycin_finder,
            error_ppm=10.0,
            max_candidates_per_peak=8,
            max_results_per_peak=500,
        )

    def test_erythromycin_tree_from_spectrum(self, erythromycin_generator):
        """Build a fragmentation tree for erythromycin [M+H]+."""
        # Synthetic MS/MS spectrum based on known erythromycin fragmentation
        spectrum = [
            (576.375, 999.0),   # loss of cladinose
            (558.365, 700.0),   # loss of cladinose + H2O
            (540.354, 300.0),   # loss of cladinose + 2×H2O
            (158.118, 600.0),   # protonated desosamine
            (116.107, 250.0),   # desosamine fragment
        ]

        tree = erythromycin_generator.generate(
            precursor_mz=self.PRECURSOR_MZ,
            spectrum=spectrum,
            charge=1,
            adduct="H",
            filter_rdbe=(0, 20),
            check_octet=True,
            max_counts=self.MAX_COUNTS,
            auto_detect_envelopes=False,
        )

        assert isinstance(tree, FragmentationTree)
        assert tree.total_score > -float("inf")
        # Should have root + at least some fragments
        assert len(tree.nodes) >= 2
        assert len(tree.edges) >= 1

    def test_erythromycin_tree_with_envelopes(self, erythromycin_generator):
        """
        Build a fragmentation tree for erythromycin using isotope envelopes.

        The envelopes provide stronger constraints for formula assignment
        compared to monoisotopic peaks alone — this is the recommended
        approach for real spectra.
        """
        envelopes = [
            # 576.375 fragment with M+0, M+1
            IsotopeEnvelope(peaks=np.array([
                [576.375, 999.0],
                [577.379, 230.0],  # ~23% M+1 (reasonable for C~30)
            ])),
            # 558.365 fragment with M+0, M+1
            IsotopeEnvelope(peaks=np.array([
                [558.365, 700.0],
                [559.368, 155.0],
            ])),
            # 158.118 protonated desosamine with M+0, M+1
            IsotopeEnvelope(peaks=np.array([
                [158.118, 600.0],
                [159.121, 55.0],
            ])),
        ]

        tree = erythromycin_generator.generate(
            precursor_mz=self.PRECURSOR_MZ,
            envelopes=envelopes,
            charge=1,
            adduct="H",
            filter_rdbe=(0, 20),
            check_octet=True,
            max_counts=self.MAX_COUNTS,
        )

        assert isinstance(tree, FragmentationTree)
        assert len(tree.nodes) >= 2

    def test_erythromycin_auto_detect_envelopes(self, erythromycin_generator):
        """
        Test auto-detection of isotope envelopes from a raw spectrum
        containing isotopologue peaks.
        """
        # Spectrum with isotopologue peaks included
        spectrum = [
            (576.375, 999.0),
            (577.379, 230.0),    # M+1 of 576.375
            (558.365, 700.0),
            (559.368, 155.0),    # M+1 of 558.365
            (158.118, 600.0),
            (159.121, 55.0),     # M+1 of 158.118
        ]

        tree = erythromycin_generator.generate(
            precursor_mz=self.PRECURSOR_MZ,
            spectrum=spectrum,
            charge=1,
            adduct="H",
            filter_rdbe=(0, 20),
            check_octet=True,
            max_counts=self.MAX_COUNTS,
            auto_detect_envelopes=True,
        )

        assert isinstance(tree, FragmentationTree)
        assert len(tree.nodes) >= 2
