from __future__ import annotations

import gzip
import math
from os import PathLike
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from molmass import Formula

from find_mfs.utils.filtering import BOND_ELECTRONS

from .default_profile import (
    COMMON_FRAGMENT_NORMALIZATION,
    COMMON_FRAGMENTS,
    COMMON_LOSS_NORMALIZATION,
    COMMON_LOSSES,
    COMMON_RADICALS,
    COMMON_ROOT_LOSS_NORMALIZATION,
    COMMON_ROOT_LOSSES,
    STRANGE_FRAGMENT_WHITELIST,
    STRANGE_LOSSES,
)

AMINO_ACID_RESIDUES = (
    "C3H5NO",
    "C3H5NOS",
    "C4H5NO3",
    "C5H7NO3",
    "C9H9NO",
    "C2H3NO",
    "C6H7N3O",
    "C6H11NO",
    "C6H12N2O",
    "C6H11NO",
    "C5H9NOS",
    "C4H6N2O2",
    "C5H7NO",
    "C5H8N2O2",
    "C6H12N4O",
    "C3H5NO2",
    "C4H7NO2",
    "C5H9NO",
    "C11H10N2O",
    "C9H9NO2",
)

SIRIUS_V6_RECOMBINED_COMMON_LOSS_OVERRIDES = {
    # SIRIUS v6 uses Trove hash-map iteration in MinimalScoreRecombinator.
    # These are the post-normalization CommonLossEdgeScorer values where that
    # order differs from the deterministic maximum recombination used here.
    "C15H21NO7": 2.4976227503252453,
    "C2H6": -1.3646611753318298,
    "C6H13NO2": -1.392815236299485,
    "C2H4O": -0.6057228211483281,
    "C5H10N2O": -0.919544447557656,
    "C4H6O": -1.3259677864895392,
    "C2H5NO": -1.461006723186391,
    "CH3NO": -0.6805871305483651,
    "C2H5N": -0.6913205656908721,
}


@dataclass(slots=True)
class SiriusLikeScoringTables:
    """
    Caller-owned SIRIUS-like lookup tables for the Rust raw-spectrum engine.

    Pass an instance to ``FragmentationTreeFinder(..., scoring_tables=tables)``
    to copy these tables into Rust once and reuse them for every spectrum call.
    """

    common_fragments: dict[str, float] = field(default_factory=dict)
    common_losses: dict[str, float] = field(default_factory=dict)
    recombined_common_losses: dict[str, float] = field(default_factory=dict)
    recombined_common_loss_overrides: dict[str, float] = field(default_factory=dict)
    common_radicals: dict[str, float] = field(default_factory=dict)
    common_root_losses: dict[str, float] = field(default_factory=dict)
    strange_fragment_whitelist: set[str] = field(default_factory=set)
    strange_losses: set[str] = field(default_factory=set)
    common_fragment_normalization: float = COMMON_FRAGMENT_NORMALIZATION
    common_loss_normalization: float = COMMON_LOSS_NORMALIZATION
    common_root_loss_normalization: float = COMMON_ROOT_LOSS_NORMALIZATION

    @classmethod
    def default(cls) -> "SiriusLikeScoringTables":
        return cls(
            common_fragments=dict(_augmented_common_fragments()),
            common_losses=dict(_augmented_common_losses()),
            recombined_common_losses=dict(_recombined_common_losses()),
            recombined_common_loss_overrides=dict(
                SIRIUS_V6_RECOMBINED_COMMON_LOSS_OVERRIDES
            ),
            common_radicals=dict(COMMON_RADICALS),
            common_root_losses=dict(COMMON_ROOT_LOSSES),
            strange_fragment_whitelist=set(STRANGE_FRAGMENT_WHITELIST),
            strange_losses=set(STRANGE_LOSSES),
            common_fragment_normalization=COMMON_FRAGMENT_NORMALIZATION,
            common_loss_normalization=COMMON_LOSS_NORMALIZATION,
            common_root_loss_normalization=COMMON_ROOT_LOSS_NORMALIZATION,
        )

    def to_rust_payload(self) -> dict[str, object]:
        return {
            "common_fragments": sorted(self.common_fragments.items()),
            "common_losses": sorted(self.common_losses.items()),
            "recombined_common_losses": sorted(self.recombined_common_losses.items()),
            "recombined_common_loss_overrides": sorted(
                self.recombined_common_loss_overrides.items()
            ),
            "common_radicals": sorted(self.common_radicals.items()),
            "common_root_losses": sorted(self.common_root_losses.items()),
            "strange_fragment_whitelist": sorted(self.strange_fragment_whitelist),
            "strange_losses": sorted(self.strange_losses),
            "common_fragment_normalization": self.common_fragment_normalization,
            "common_loss_normalization": self.common_loss_normalization,
            "common_root_loss_normalization": self.common_root_loss_normalization,
        }


@dataclass(frozen=True, slots=True)
class SiriusLikeScoringConfig:
    """
    Hard-coded default SIRIUS-like scoring constants.

    This intentionally does not expose custom profile JSON loading. Built-in
    SIRIUS default-profile lookup tables are vendored as constants.
    DB-paired scoring is disabled unless `db_paired_formulas` is supplied,
    because SIRIUS stores that formula map as a packaged runtime resource.
    """

    ms2_tolerance_ppm: float = 10.0
    candidate_search_ppm: float = 15.0
    candidate_search_absolute_da: float = 0.003
    precursor_tolerance_ppm: float = 10.0
    candidate_limit_per_peak: int = 20
    max_fragment_peaks: int = 59
    min_relative_intensity: float = 0.0
    merge_close_peaks: bool = True
    median_noise_intensity: float = 0.015
    tree_size_score: float = -0.5
    fragment_size_max_score: float = 2.0
    fragment_size_max_mz: float = 200.0
    clipped_noise_xmin: float = 0.002
    clipped_noise_beta: float = 0.00001
    loss_size_mean: float = 4.022526672023266
    loss_size_variance: float = 0.3124649410213113
    loss_size_normalization: float = -5.310349962255842
    intrinsically_charged_root_penalty: float = -4.605170185988091
    strange_element_root_penalty: float = -1.6094379124341003
    strange_element_small_fragment_score: float = 0.5
    strange_element_small_fragment_max_mass: float = 75.0
    strange_element_fragment_score: float = 0.693147
    strange_element_fragment_penalty: float = -0.4054652081081694
    strange_element_fragment_min_mass: float = 100.0
    strange_element_loss_score: float = 0.6931471805599453
    free_radical_penalty: float = -2.3025850929940455
    free_radical_normalization: float = -0.011626542158820332
    strict_sirius_radical_parity: bool = False
    dbe_loss_score: float = -1.0986122886681098
    pure_carbon_nitrogen_loss_penalty: float = -2.3025850929940455
    mass_deviation_vertex_weight: float = 0.5
    mass_deviation_edge_weight: float = 0.5
    mass_deviation_absolute_da: float = 0.002
    loss_mass_deviation_absolute_da: float = 0.001
    chemical_prior_root_score: float = 1.0
    db_paired_formula_score: float = 1.0
    db_paired_formulas: frozenset[str] | None = None
    enable_common_fragment_score: bool = True
    carbohydrogen_root_score: float = 2.5
    enable_carbohydrogen_fragment_score: bool = False
    carbohydrogen_fragment_min_relative_intensity: float = 0.02
    carbohydrogen_fragment_xmin: float = 0.02
    carbohydrogen_fragment_median: float = 0.5
    multimere_root_loss_score: float = 10.0
    multimere_loss_score: float = 2.0
    fatty_acid_chain_score_weight: float = 0.5
    fatty_acid_chain_double_bond_decay: float = 0.95
    fatty_acid_chain_min_length: int = 6
    fatty_acid_chain_max_length: int = 36
    fatty_acid_chain_max_double_bonds: int = 6
    recombine_common_losses: bool = True
    estimate_tree_size: bool = True
    tree_size_increase: float = 1.0
    max_tree_size_increase: float = 3.0
    max_tree_size_score: float = 2.5
    min_explained_intensity: float = 0.7
    min_explained_peaks: int = 15
    use_sirius_tree_size_quality_threshold: bool = False

    @classmethod
    def sirius_v6_reference(
        cls,
        *,
        db_paired_formulas: frozenset[str] | None = None,
    ) -> "SiriusLikeScoringConfig":
        """
        Return the closest built-in SIRIUS v6 reference profile.

        The packaged SIRIUS DB formula map is intentionally not bundled. Pass a
        caller-owned formula set, for example from `load_db_paired_formulas`,
        when reproducing SIRIUS v6 validation runs.
        """

        return cls(
            strict_sirius_radical_parity=True,
            enable_common_fragment_score=False,
            enable_carbohydrogen_fragment_score=True,
            db_paired_formulas=db_paired_formulas,
            use_sirius_tree_size_quality_threshold=True,
        )


class SiriusLikeScorer:
    """Computes scalar SIRIUS-like scoring terms for one spectrum."""

    def __init__(self, symbols: Sequence[str], config: SiriusLikeScoringConfig):
        self.symbols = list(symbols)
        self.config = config

    def root_score(
        self,
        formula: str,
        counts: Sequence[int],
        observed_mz: float,
        theoretical_mz: float,
        ppm: float | None = None,
    ) -> float:
        return (
            self.mass_deviation_score(
                observed_mz,
                theoretical_mz,
                self.config.precursor_tolerance_ppm if ppm is None else ppm,
                self.config.mass_deviation_vertex_weight,
            )
            + self.intrinsically_charged_root_score(counts)
            + self.phosphor_root_score(counts)
            + self.strange_element_root_score(counts)
            + self.config.chemical_prior_root_score
            + self.carbohydrogen_root_score(counts)
            + self.db_paired_score(formula)
        )

    def fragment_candidate_score(
        self,
        formula: str,
        counts: Sequence[int],
        observed_mz: float,
        theoretical_mz: float,
        neutral_mass: float,
        relative_intensity: float = 0.0,
    ) -> float:
        return (
            self.mass_deviation_score(
                observed_mz,
                theoretical_mz,
                self.config.ms2_tolerance_ppm,
                self.config.mass_deviation_vertex_weight,
            )
            + self.phosphor_fragment_score(counts)
            + self.strange_element_small_fragment_score(counts, neutral_mass)
            + self.strange_element_fragment_score(formula, counts, neutral_mass)
            + self.common_fragment_score(formula)
            + self.carbohydrogen_fragment_score(counts, relative_intensity)
        )

    def peak_score(self, mz: float, relative_intensity: float) -> float:
        return (
            self.clipped_peak_is_noise_score(relative_intensity)
            + self.config.tree_size_score
            + self.fragment_size_score(mz)
        )

    def peak_pair_score(self, parent_mz: float, child_mz: float) -> float:
        delta = parent_mz - child_mz
        if delta <= 0:
            return 0.0
        return self.loss_size_score(delta)

    def loss_score(
        self,
        loss_formula: str,
        loss_counts: Sequence[int],
        observed_delta: float,
        theoretical_delta: float,
        child_formula: str | None = None,
        is_root_loss: bool = False,
    ) -> float:
        return (
            self.mass_deviation_score(
                observed_delta,
                theoretical_delta,
                self.config.ms2_tolerance_ppm,
                self.config.mass_deviation_edge_weight,
                self.config.loss_mass_deviation_absolute_da,
            )
            + self.phosphor_fragment_score(loss_counts)
            + self.free_radical_loss_score(loss_formula, loss_counts)
            + self.dbe_loss_score(loss_counts)
            + self.pure_carbon_nitrogen_loss_score(loss_counts)
            + self.strange_element_loss_score(loss_formula)
            + self.common_loss_score(loss_formula)
            + self.multimere_loss_score(loss_formula, child_formula, is_root_loss)
            + self.fatty_acid_chain_loss_score(loss_formula, loss_counts)
        )

    def mass_deviation_score(
        self,
        observed_mz: float,
        theoretical_mz: float,
        ppm: float,
        weight: float = 1.0,
        absolute_da: float | None = None,
    ) -> float:
        absolute = (
            self.config.mass_deviation_absolute_da
            if absolute_da is None
            else absolute_da
        )
        sigma = max(abs(observed_mz) * ppm * 1e-6, absolute)
        if sigma <= 0.0 or not math.isfinite(sigma):
            return -100.0
        x = abs(observed_mz - theoretical_mz) / (math.sqrt(2.0) * sigma)
        prob = max(math.erfc(x), 1e-300)
        return max(weight * math.log(prob), -100.0)

    def clipped_peak_is_noise_score(self, relative_intensity: float) -> float:
        if relative_intensity <= 0.0:
            return 0.0
        cdf_one = self._pareto_cdf(1.0)
        c = 1.0 - cdf_one
        q = (
            1.0
            - self._pareto_cdf(min(relative_intensity, 1.0))
            - c
            + self.config.clipped_noise_beta
        ) / (1.0 - c + self.config.clipped_noise_beta)
        return -math.log(max(q, 1e-300))

    def fragment_size_score(self, mz: float) -> float:
        fraction = min(1.0, max(0.0, mz) / self.config.fragment_size_max_mz)
        return self.config.fragment_size_max_score * (1.0 - fraction)

    def loss_size_score(self, mass: float) -> float:
        if mass <= 0.0:
            return -100.0
        variance = self.config.loss_size_variance
        sd = math.sqrt(variance)
        density = (
            math.exp(-((math.log(mass) - self.config.loss_size_mean) ** 2) / (2.0 * variance))
            / (math.sqrt(2.0 * math.pi) * sd * mass)
        )
        return math.log(max(1e-12, density)) - self.config.loss_size_normalization

    def intrinsically_charged_root_score(self, counts: Sequence[int]) -> float:
        if self.maybe_charged_counts(counts):
            return self.config.intrinsically_charged_root_penalty
        return 0.0

    def phosphor_root_score(self, counts: Sequence[int]) -> float:
        p = self.count_of(counts, "P")
        if p <= 0:
            return 0.0
        if self.count_of(counts, "O") + self.count_of(counts, "S") < 2 * p:
            return math.log(0.05)
        return 0.0

    def phosphor_fragment_score(self, counts: Sequence[int]) -> float:
        p = self.count_of(counts, "P")
        if p > 0 and self.count_of(counts, "O") < p and self.count_of(counts, "S") < p:
            return math.log(0.25)
        return 0.0

    def strange_element_root_score(self, counts: Sequence[int]) -> float:
        n_strange = sum(
            1
            for symbol, count in zip(self.symbols, counts)
            if count > 0 and symbol not in {"C", "H", "N", "O"}
        )
        return self.config.strange_element_root_penalty * n_strange

    def strange_element_small_fragment_score(
        self,
        counts: Sequence[int],
        neutral_mass: float,
    ) -> float:
        if neutral_mass > self.config.strange_element_small_fragment_max_mass:
            return 0.0
        if any(
            count > 0 and symbol not in {"C", "H", "N", "O"}
            for symbol, count in zip(self.symbols, counts)
        ):
            return self.config.strange_element_small_fragment_score
        return 0.0

    def strange_element_fragment_score(
        self,
        formula: str,
        counts: Sequence[int],
        neutral_mass: float,
    ) -> float:
        if formula in STRANGE_FRAGMENT_WHITELIST:
            return self.config.strange_element_fragment_score
        if neutral_mass < self.config.strange_element_fragment_min_mass:
            return 0.0
        if any(
            count > 0 and symbol not in {"C", "H", "N", "O"}
            for symbol, count in zip(self.symbols, counts)
        ):
            return self.config.strange_element_fragment_penalty
        return 0.0

    def common_fragment_score(self, formula: str) -> float:
        if not self.config.enable_common_fragment_score:
            return 0.0
        score = _augmented_common_fragments().get(formula)
        if score is None:
            return 0.0
        return score - COMMON_FRAGMENT_NORMALIZATION

    def free_radical_loss_score(self, formula: str, counts: Sequence[int]) -> float:
        if formula in COMMON_RADICALS:
            return COMMON_RADICALS[formula] - self.config.free_radical_normalization
        if self.maybe_charged_counts(counts):
            return self.config.free_radical_penalty - self.config.free_radical_normalization
        return -self.config.free_radical_normalization

    def dbe_loss_score(self, counts: Sequence[int]) -> float:
        doubled = self.doubled_rdbe(counts)
        if doubled is not None and doubled < 0:
            return max(math.log(0.05), abs(doubled) * self.config.dbe_loss_score)
        return 0.0

    def pure_carbon_nitrogen_loss_score(self, counts: Sequence[int]) -> float:
        total = sum(counts)
        if total <= 0:
            return 0.0
        cn = self.count_of(counts, "C") + self.count_of(counts, "N")
        if cn >= total:
            return self.config.pure_carbon_nitrogen_loss_penalty
        return 0.0

    def strange_element_loss_score(self, formula: str) -> float:
        if formula in STRANGE_LOSSES:
            return self.config.strange_element_loss_score
        return 0.0

    def common_loss_score(self, formula: str) -> float:
        if self.config.recombine_common_losses:
            override = SIRIUS_V6_RECOMBINED_COMMON_LOSS_OVERRIDES.get(formula)
            if override is not None:
                return override
            score = _recombined_common_losses().get(formula, 0.0)
            if score != 0.0:
                return score - COMMON_LOSS_NORMALIZATION
        score = _augmented_common_losses().get(formula)
        if score is None:
            return -COMMON_LOSS_NORMALIZATION
        return score - COMMON_LOSS_NORMALIZATION

    def common_root_loss_score(self, formula: str) -> float:
        score = COMMON_ROOT_LOSSES.get(formula)
        if score is None:
            return -COMMON_ROOT_LOSS_NORMALIZATION
        return score - COMMON_ROOT_LOSS_NORMALIZATION

    def db_paired_score(self, formula: str, root_formula: str | None = None) -> float:
        formulas = self.config.db_paired_formulas
        if formulas is None or formula not in formulas:
            return 0.0
        if root_formula is not None and root_formula not in formulas:
            return 0.0
        return self.config.db_paired_formula_score

    def multimere_loss_score(
        self,
        loss_formula: str,
        child_formula: str | None,
        is_root_loss: bool,
    ) -> float:
        if child_formula is None or loss_formula != child_formula:
            return 0.0
        if is_root_loss:
            return self.config.multimere_root_loss_score
        return self.config.multimere_loss_score

    def fatty_acid_chain_loss_score(
        self,
        loss_formula: str,
        counts: Sequence[int],
    ) -> float:
        chain = self.lipid_chain_from_counts(counts)
        if chain is None:
            return 0.0
        chain_length, double_bonds = chain
        if (
            chain_length < self.config.fatty_acid_chain_min_length
            or chain_length > self.config.fatty_acid_chain_max_length
            or double_bonds > self.config.fatty_acid_chain_max_double_bonds
        ):
            return 0.0
        penalty = self.loss_size_score(neutral_mass(loss_formula))
        if penalty >= 0.0:
            return 0.0
        return (
            -penalty
            * self.config.fatty_acid_chain_score_weight
            * (
                self.config.fatty_acid_chain_double_bond_decay
                ** (double_bonds * double_bonds)
            )
        )

    def lipid_chain_from_counts(self, counts: Sequence[int]) -> tuple[int, int] | None:
        if any(
            count > 0 and symbol not in {"C", "H", "N", "O"}
            for symbol, count in zip(self.symbols, counts)
        ):
            return None

        c = self.count_of(counts, "C")
        h = self.count_of(counts, "H")
        n = self.count_of(counts, "N")
        o = self.count_of(counts, "O")
        if c < 2:
            return None

        if n > 0:
            if n == 1 and o == 2 and h % 2 != 0:
                double_bonds = ((c * 2 + 3) - h) // 2
                if double_bonds >= c // 2:
                    return None
                if double_bonds >= 0:
                    return c, double_bonds
        elif o > 0 and h % 2 == 0:
            if o == 1:
                double_bonds = ((c * 2 - 2) - h) // 2
                if double_bonds >= c // 2:
                    return None
                if double_bonds >= 0:
                    return c, double_bonds
        elif h % 2 == 0:
            double_bonds = (2 * c - h) // 2
            if double_bonds >= c // 2:
                return None
            if double_bonds >= 0:
                return c, double_bonds
        return None

    def carbohydrogen_root_score(self, counts: Sequence[int]) -> float:
        if self.is_cho_counts(counts):
            return self.config.carbohydrogen_root_score
        return 0.0

    def carbohydrogen_fragment_score(
        self,
        counts: Sequence[int],
        relative_intensity: float,
    ) -> float:
        if (
            not self.config.enable_carbohydrogen_fragment_score
            or
            relative_intensity <= self.config.carbohydrogen_fragment_min_relative_intensity
            or not self.is_cho_counts(counts)
        ):
            return 0.0
        return self._pareto_cdf_from_median(
            relative_intensity,
            self.config.carbohydrogen_fragment_xmin,
            self.config.carbohydrogen_fragment_median,
        )

    def count_of(self, counts: Sequence[int], symbol: str) -> int:
        try:
            return int(counts[self.symbols.index(symbol)])
        except ValueError:
            return 0

    def is_cho_counts(self, counts: Sequence[int]) -> bool:
        return all(
            count <= 0 or symbol in {"C", "H", "O"}
            for symbol, count in zip(self.symbols, counts)
        )

    def doubled_rdbe(self, counts: Sequence[int]) -> int | None:
        total = 2
        for symbol, count in zip(self.symbols, counts):
            valence = BOND_ELECTRONS.get(symbol)
            if valence is None:
                return None
            total += int(count) * (valence - 2)
        return total

    def maybe_charged_counts(self, counts: Sequence[int]) -> bool:
        doubled = self.doubled_rdbe(counts)
        if doubled is None:
            return False
        if self.config.strict_sirius_radical_parity:
            return doubled > 0 and doubled % 2 == 1
        return abs(doubled) % 2 == 1

    def _pareto_cdf(self, x: float) -> float:
        xmin = self.config.clipped_noise_xmin
        if x < xmin:
            return 0.0
        median = max(self.config.median_noise_intensity, xmin * (1.0 + 1e-9))
        k = math.log(2.0) / math.log(median / xmin)
        return 1.0 - (xmin / x) ** k

    @staticmethod
    def _pareto_cdf_from_median(x: float, xmin: float, median: float) -> float:
        if x < xmin:
            return 0.0
        k = math.log(2.0) / math.log(median / xmin)
        return 1.0 - (xmin / x) ** k


@lru_cache(maxsize=4096)
def neutral_mass(formula: str) -> float:
    return Formula(formula).monoisotopic_mass


@lru_cache(maxsize=4096)
def protonated_mz(formula: str) -> float:
    return Formula(formula).monoisotopic_mass + Formula("H+").monoisotopic_mass


def load_db_paired_formulas(path: str | PathLike[str]) -> frozenset[str]:
    """
    Load caller-owned DB-paired formula identifiers for SIRIUS-like scoring.

    The file format is intentionally simple: one formula per line, with blank
    lines and `#` comments ignored. Comma- or whitespace-separated rows are
    accepted by taking the first field, so exported CSV/text formula lists can
    be used directly. `.gz` files are decompressed automatically.
    """

    path_text = str(path)
    opener = gzip.open if path_text.endswith(".gz") else open
    formulas = set()
    with opener(path, "rt") as handle:
        for line in handle:
            clean = line.partition("#")[0].strip()
            if not clean:
                continue
            formula = clean.replace(",", " ").split()[0]
            if formula:
                formulas.add(formula)
    return frozenset(formulas)


@lru_cache(maxsize=1)
def _recombined_common_losses() -> dict[str, float]:
    recombined: dict[str, float] = {}
    source = _augmented_common_losses()
    formulas = list(source)
    config = SiriusLikeScoringConfig(recombine_common_losses=False)
    scorer = SiriusLikeScorer((), config)

    for i, first in enumerate(formulas):
        first_common_score = source[first]
        if first_common_score < 0.0:
            continue
        first_score = scorer.loss_size_score(neutral_mass(first)) + first_common_score
        for second in formulas[i:]:
            second_common_score = source[second]
            if second_common_score < 0.0:
                continue
            second_score = scorer.loss_size_score(neutral_mass(second)) + second_common_score
            combined = _add_formula_strings(first, second)
            combined_loss_size = scorer.loss_size_score(neutral_mass(combined))
            combined_score = combined_loss_size + source.get(combined, 0.0)
            recombination_score = min(first_score, second_score) - 1.0
            if recombination_score > combined_score:
                final_score = recombination_score - combined_loss_size
                recombined[combined] = max(
                    final_score,
                    recombined.get(combined, -math.inf),
                )
    return recombined


@lru_cache(maxsize=1)
def _augmented_common_losses() -> dict[str, float]:
    common = dict(COMMON_LOSSES)
    config = SiriusLikeScoringConfig(recombine_common_losses=False)
    scorer = SiriusLikeScorer((), config)
    for residue in AMINO_ACID_RESIDUES:
        score = 1.0 - min(0.0, scorer.loss_size_score(neutral_mass(residue)))
        common[residue] = max(score, common.get(residue, -math.inf))
    return common


@lru_cache(maxsize=1)
def _augmented_common_fragments() -> dict[str, float]:
    common = dict(COMMON_FRAGMENTS)
    for residue in AMINO_ACID_RESIDUES:
        common[residue] = max(0.5, common.get(residue, -math.inf))
        hydrated = _add_formula_strings(residue, "H2O")
        common[hydrated] = max(0.5, common.get(hydrated, -math.inf))
    return common


@lru_cache(maxsize=8192)
def _formula_counts(formula: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (symbol, int(item.count))
        for symbol, item in Formula(formula).composition().items()
    )


@lru_cache(maxsize=16384)
def _add_formula_strings(first: str, second: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for formula in (first, second):
        for symbol, count in _formula_counts(formula):
            counts[symbol] += count
    return _format_formula_counts(counts)


def _format_formula_counts(counts: dict[str, int]) -> str:
    symbols = []
    if counts.get("C", 0) > 0:
        symbols.append("C")
    if counts.get("H", 0) > 0:
        symbols.append("H")
    symbols.extend(
        sorted(
            symbol
            for symbol, count in counts.items()
            if count > 0 and symbol not in {"C", "H"}
        )
    )
    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}"
        for symbol in symbols
    )
