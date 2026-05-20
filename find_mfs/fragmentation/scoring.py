from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SiriusLikeScoringConfig:
    """
    Hard-coded default SIRIUS-like scoring constants.

    This intentionally does not expose custom profile JSON loading. Built-in
    SIRIUS default-profile lookup tables are vendored as constants.
    """

    ms2_tolerance_ppm: float = 10.0
    candidate_search_ppm: float = 15.0
    precursor_tolerance_ppm: float = 10.0
    candidate_limit_per_peak: int = 20
    max_fragment_peaks: int = 60
    min_relative_intensity: float = 0.0
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
    dbe_loss_score: float = -1.0986122886681098
    pure_carbon_nitrogen_loss_penalty: float = -2.3025850929940455
    mass_deviation_vertex_weight: float = 1.0
    mass_deviation_edge_weight: float = 0.5
    mass_deviation_absolute_da: float = 0.0
    loss_mass_deviation_absolute_da: float = 0.001


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
    ) -> float:
        return (
            self.mass_deviation_score(
                observed_mz,
                theoretical_mz,
                self.config.precursor_tolerance_ppm,
                self.config.mass_deviation_vertex_weight,
            )
            + self.intrinsically_charged_root_score(counts)
            + self.phosphor_root_score(counts)
            + self.strange_element_root_score(counts)
        )

    def fragment_candidate_score(
        self,
        formula: str,
        counts: Sequence[int],
        observed_mz: float,
        theoretical_mz: float,
        neutral_mass: float,
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
            + self.common_root_loss_score(loss_formula, is_root_loss)
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
        doubled = self.doubled_rdbe(counts)
        if doubled is not None and abs(round(doubled) % 2) == 1:
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
        score = COMMON_FRAGMENTS.get(formula)
        if score is None:
            return 0.0
        return score - COMMON_FRAGMENT_NORMALIZATION

    def free_radical_loss_score(self, formula: str, counts: Sequence[int]) -> float:
        if formula in COMMON_RADICALS:
            return COMMON_RADICALS[formula] - self.config.free_radical_normalization
        doubled = self.doubled_rdbe(counts)
        if doubled is not None and abs(round(doubled) % 2) == 1:
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
        score = COMMON_LOSSES.get(formula)
        if score is None:
            return -COMMON_LOSS_NORMALIZATION
        return score - COMMON_LOSS_NORMALIZATION

    def common_root_loss_score(self, formula: str, is_root_loss: bool) -> float:
        if not is_root_loss:
            return 0.0
        score = COMMON_ROOT_LOSSES.get(formula)
        if score is None:
            return -COMMON_ROOT_LOSS_NORMALIZATION
        return score - COMMON_ROOT_LOSS_NORMALIZATION

    def count_of(self, counts: Sequence[int], symbol: str) -> int:
        try:
            return int(counts[self.symbols.index(symbol)])
        except ValueError:
            return 0

    def doubled_rdbe(self, counts: Sequence[int]) -> int | None:
        total = 2
        for symbol, count in zip(self.symbols, counts):
            valence = BOND_ELECTRONS.get(symbol)
            if valence is None:
                return None
            total += int(count) * (valence - 2)
        return total

    def _pareto_cdf(self, x: float) -> float:
        xmin = self.config.clipped_noise_xmin
        if x < xmin:
            return 0.0
        median = max(self.config.median_noise_intensity, xmin * (1.0 + 1e-9))
        k = math.log(2.0) / math.log(median / xmin)
        return 1.0 - (xmin / x) ** k


@lru_cache(maxsize=4096)
def neutral_mass(formula: str) -> float:
    return Formula(formula).monoisotopic_mass


@lru_cache(maxsize=4096)
def protonated_mz(formula: str) -> float:
    return Formula(formula).monoisotopic_mass + Formula("H+").monoisotopic_mass
