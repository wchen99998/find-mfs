from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from collections.abc import Iterable, Sequence

from find_mfs.core.finder import FormulaFinder

from .results import (
    ExplicitFragmentationScoring,
    Fragment,
    FragmentCandidate,
    FragmentationTree,
    FragmentationTreeOptions,
    Loss,
    SpectrumPeak,
)
from .scoring import (
    SiriusLikeScorer,
    SiriusLikeScoringConfig,
    protonated_mz,
)
from .spectrum import FragmentationSpectrum


@dataclass(frozen=True, slots=True)
class _GeneratedCandidate:
    candidate: FragmentCandidate
    counts: tuple[int, ...]
    theoretical_mz: float


class FragmentationTreeFinder:
    """
    Public API for SIRIUS-style fragmentation tree optimization.

    The current implementation uses the Rust backend and solves the exact
    colorful subtree ILP with HiGHS. Candidate generation and scoring can be
    kept explicit by passing ``FragmentCandidate`` values and an
    ``ExplicitFragmentationScoring`` object.
    """

    def __init__(
        self,
        elements: Iterable[str] | str = "CHNOPS",
        backend: str = "rust",
    ):
        if backend != "rust":
            raise ValueError(
                f"Unknown backend {backend!r}. Fragmentation trees currently "
                "support only the 'rust' backend."
            )
        self.backend = backend
        self.element_symbols = self._normalize_element_symbols(elements)

    def find_tree(
        self,
        root_candidates: Sequence[FragmentCandidate],
        fragment_candidates: Sequence[FragmentCandidate],
        scoring: ExplicitFragmentationScoring | None = None,
        options: FragmentationTreeOptions | None = None,
        allowed_ionizations: Sequence[str] | None = None,
    ) -> FragmentationTree:
        return self.solve_tree(
            root_candidates=root_candidates,
            fragment_candidates=fragment_candidates,
            scoring=scoring,
            options=options,
            allowed_ionizations=allowed_ionizations,
        )

    def find_tree_from_spectrum(
        self,
        spectrum: FragmentationSpectrum,
        scoring_config: SiriusLikeScoringConfig | None = None,
        options: FragmentationTreeOptions | None = None,
    ) -> FragmentationTree:
        """
        Generate candidates and SIRIUS-like scores from a raw MS/MS spectrum.

        This uses hard-coded default SIRIUS-like scalar scorers and the existing
        formula finder for peak decompositions. It intentionally does not load
        custom SIRIUS profile JSON files.
        """
        if spectrum.precursor_formula is None:
            raise ValueError("spectrum.precursor_formula is required")
        if spectrum.precursor_ion != "[M+H]+":
            raise ValueError(
                "raw spectrum fragmentation trees currently support only [M+H]+"
            )

        config = SiriusLikeScoringConfig() if scoring_config is None else scoring_config
        if config.estimate_tree_size:
            tree = self._find_tree_from_spectrum_with_tree_size_estimation(
                spectrum,
                config,
                options,
            )
        else:
            tree = self._find_tree_from_spectrum_once(spectrum, config, options)
        tree.query_params.update(
            {
                "spectrum_name": spectrum.name,
                "spectrum_accession": spectrum.accession,
                "precursor_mz": spectrum.precursor_mz,
                "precursor_formula": spectrum.precursor_formula,
                "precursor_ion": spectrum.precursor_ion,
                "scoring": "sirius_like_default",
            }
        )
        return tree

    def _find_tree_from_spectrum_with_tree_size_estimation(
        self,
        spectrum: FragmentationSpectrum,
        config: SiriusLikeScoringConfig,
        options: FragmentationTreeOptions | None,
    ) -> FragmentationTree:
        tree_size = config.tree_size_score
        increase = 0.0
        last_tree = None
        last_tree_size = tree_size
        while increase <= config.max_tree_size_increase:
            current_config = replace(config, tree_size_score=tree_size)
            tree, generated, processed_peak_count = self._find_tree_from_spectrum_once(
                spectrum,
                current_config,
                options,
                include_generated=True,
            )
            last_tree = tree
            last_tree_size = tree_size
            if self._is_high_quality_tree(
                tree,
                generated,
                config,
                processed_peak_count,
            ):
                break
            increase += config.tree_size_increase
            tree_size += config.tree_size_increase
            if tree_size > config.max_tree_size_score:
                break
        if last_tree is None:
            raise ValueError("no fragmentation tree could be computed")
        last_tree.query_params["tree_size_score"] = last_tree_size
        return last_tree

    def _find_tree_from_spectrum_once(
        self,
        spectrum: FragmentationSpectrum,
        config: SiriusLikeScoringConfig,
        options: FragmentationTreeOptions | None,
        include_generated: bool = False,
    ):
        scorer = SiriusLikeScorer(self.element_symbols, config)
        root_counts = self.parse_formula_counts(spectrum.precursor_formula)
        root_peak, fragment_peaks = self._split_root_peak(spectrum, config)
        processed_peak_count = len(fragment_peaks) + 1
        intensity_scale = self._processed_intensity_scale(root_peak, fragment_peaks)
        root_mz = root_peak.mz if root_peak is not None else spectrum.precursor_mz
        root_intensity = root_peak.intensity if root_peak is not None else None
        root_theoretical_mz = protonated_mz(spectrum.precursor_formula)
        root_candidate = FragmentCandidate(
            formula=spectrum.precursor_formula,
            mass=root_mz,
            score=scorer.root_score(
                spectrum.precursor_formula,
                root_counts,
                root_mz if root_peak is not None else root_theoretical_mz,
                root_theoretical_mz,
                config.ms2_tolerance_ppm,
            ),
            peak_id=0,
            color=0,
            ionization=spectrum.precursor_ion,
            intensity=root_intensity,
        )

        generated = self._generate_fragment_candidates(
            fragment_peaks,
            root_counts,
            config,
            scorer,
            spectrum.precursor_formula,
            spectrum.precursor_ion,
            intensity_scale,
        )
        fragment_candidates = [item.candidate for item in generated.values()]
        root_generated = _GeneratedCandidate(
            candidate=root_candidate,
            counts=root_counts,
            theoretical_mz=root_theoretical_mz,
        )
        scoring = self._build_sirius_like_scoring(
            root_generated,
            generated,
            scorer,
            intensity_scale,
        )

        tree = self.find_tree(
            [root_candidate],
            fragment_candidates,
            scoring=scoring,
            options=options,
            allowed_ionizations=[spectrum.precursor_ion],
        )
        tree.query_params["tree_size_score"] = config.tree_size_score
        if include_generated:
            return tree, generated, processed_peak_count
        return tree

    def _is_high_quality_tree(
        self,
        tree: FragmentationTree,
        generated: dict[str, _GeneratedCandidate],
        config: SiriusLikeScoringConfig,
        processed_peak_count: int,
    ) -> bool:
        explainable_by_color = {}
        for item in generated.values():
            intensity = item.candidate.intensity or 0.0
            explainable_by_color[item.candidate.tree_color] = max(
                intensity,
                explainable_by_color.get(item.candidate.tree_color, 0.0),
            )
        if not explainable_by_color:
            return False

        explained_colors = {
            fragment.color
            for fragment in tree.fragments
            if fragment.color in explainable_by_color
        }
        total_intensity = sum(explainable_by_color.values())
        explained_intensity = sum(
            explainable_by_color[color]
            for color in explained_colors
        )
        intensity_ratio = (
            0.0 if total_intensity <= 0.0 else explained_intensity / total_intensity
        )
        if config.use_sirius_tree_size_quality_threshold:
            # SIRIUS uses the processed merged-peak count, including the parent
            # peak, and requires tree vertices >= min(mergedPeaks - 2, 15).
            min_vertices = min(
                max(0, processed_peak_count - 2),
                config.min_explained_peaks,
            )
        else:
            min_vertices = min(
                len(explainable_by_color) + 1,
                config.min_explained_peaks,
            )
        return (
            intensity_ratio >= config.min_explained_intensity
            and len(tree.fragments) >= min_vertices
        )

    @staticmethod
    def _processed_intensity_scale(
        root_peak: SpectrumPeak | None,
        fragment_peaks: Sequence[SpectrumPeak],
    ) -> float:
        intensities = [peak.intensity for peak in fragment_peaks]
        if root_peak is not None:
            intensities.append(root_peak.intensity)
        return max(intensities, default=1.0)

    def solve_tree(
        self,
        root_candidates: Sequence[FragmentCandidate],
        fragment_candidates: Sequence[FragmentCandidate],
        scoring: ExplicitFragmentationScoring | None = None,
        options: FragmentationTreeOptions | None = None,
        allowed_ionizations: Sequence[str] | None = None,
    ) -> FragmentationTree:
        if not root_candidates:
            raise ValueError("at least one root candidate is required")

        scoring = ExplicitFragmentationScoring() if scoring is None else scoring
        options = FragmentationTreeOptions() if options is None else options
        rust = self._rust_module()
        candidates_by_formula = self._candidate_index(
            root_candidates,
            fragment_candidates,
        )

        root_payload = self._candidate_payload(root_candidates)
        fragment_payload = self._candidate_payload(fragment_candidates)

        (
            tree_score,
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
            root_payload,
            fragment_payload,
            allowed_ionizations=list(allowed_ionizations)
            if allowed_ionizations is not None
            else None,
            peak_scores=list(scoring.peak_scores.items()) or None,
            peak_pair_scores=[
                (parent_color, child_color, score)
                for (parent_color, child_color), score in scoring.peak_pair_scores.items()
            ]
            or None,
            fragment_scores=list(scoring.fragment_scores.items()) or None,
            loss_scores=[
                (parent_formula, child_formula, score)
                for (parent_formula, child_formula), score in scoring.loss_scores.items()
            ]
            or None,
            general_graph_score=scoring.general_graph_score,
            reduce_graph=options.reduce_graph,
            minimal_score=options.minimal_score,
            time_limit_seconds=options.time_limit_seconds,
            threads=options.threads,
        )

        fragments_by_formula = {}
        fragments = []
        for formula in selected_formulas:
            candidate = candidates_by_formula.get(formula)
            if candidate is None:
                raise ValueError(
                    "Rust solver selected formula "
                    f"{formula!r}, but it was not present in the input candidates"
                )
            fragment = Fragment(
                formula=formula,
                counts=self.parse_formula_counts(formula),
                ionization=candidate.ionization,
                peak_id=candidate.peak_id,
                color=candidate.tree_color,
                mass=candidate.mass,
                candidate_score=candidate.score,
                intensity=candidate.intensity,
            )
            fragments.append(fragment)
            fragments_by_formula.setdefault(formula, fragment)

        root = fragments_by_formula[root_formula]
        losses = []
        for source_formula, target_formula, score in selected_losses:
            if not source_formula:
                continue
            source = fragments_by_formula[source_formula]
            target = fragments_by_formula[target_formula]
            losses.append(
                Loss(
                    source=source,
                    target=target,
                    formula=self.format_formula_counts(
                        self.subtract_counts(source.counts, target.counts)
                    ),
                    score=score,
                )
            )

        return FragmentationTree(
            root=root,
            fragments=fragments,
            losses=losses,
            tree_score=tree_score,
            is_optimal=is_optimal,
            solver_status=status,
            graph_vertex_count=graph_vertices,
            graph_edge_count=graph_edges,
            reduced_vertex_count=reduced_vertices,
            reduced_edge_count=reduced_edges,
            query_params={
                "elements": tuple(self.element_symbols),
                "backend": self.backend,
                "allowed_ionizations": (
                    None if allowed_ionizations is None else tuple(allowed_ionizations)
                ),
                "reduce_graph": options.reduce_graph,
                "minimal_score": options.minimal_score,
                "time_limit_seconds": options.time_limit_seconds,
                "threads": options.threads,
            },
        )

    def parse_formula_counts(self, formula: str) -> tuple[int, ...]:
        rust = self._rust_module()
        return tuple(map(int, rust.parse_formula_counts(formula, self.element_symbols)))

    def format_formula_counts(self, counts: Sequence[int], charge: int = 0) -> str:
        rust = self._rust_module()
        return rust.format_formula(self.element_symbols, list(counts), charge)

    def is_subformula(self, parent_formula: str, child_formula: str) -> bool:
        parent = self.parse_formula_counts(parent_formula)
        child = self.parse_formula_counts(child_formula)
        return self.is_subformula_counts(parent, child)

    @staticmethod
    def is_subformula_counts(parent: Sequence[int], child: Sequence[int]) -> bool:
        return len(parent) == len(child) and all(
            parent_count >= child_count
            for parent_count, child_count in zip(parent, child)
        )

    @staticmethod
    def subtract_counts(parent: Sequence[int], child: Sequence[int]) -> tuple[int, ...]:
        if not FragmentationTreeFinder.is_subformula_counts(parent, child):
            raise ValueError("child counts must be a subformula of parent counts")
        return tuple(
            parent_count - child_count
            for parent_count, child_count in zip(parent, child)
        )

    def _split_root_peak(
        self,
        spectrum: FragmentationSpectrum,
        config: SiriusLikeScoringConfig,
    ) -> tuple[SpectrumPeak | None, list[SpectrumPeak]]:
        peaks = [
            peak
            for peak in sorted(spectrum.peaks, key=lambda item: item.mz)
            if peak.intensity > 0.0
        ]
        if not peaks:
            return None, []
        if config.merge_close_peaks:
            peaks = self._remove_close_lower_intensity_peaks(peaks, config)

        parent_merge_tolerance = max(
            spectrum.precursor_mz * config.ms2_tolerance_ppm * 2e-6,
            config.mass_deviation_absolute_da * 2.0,
        )
        parent_window = [
            peak
            for peak in peaks
            if abs(peak.mz - spectrum.precursor_mz) <= parent_merge_tolerance
        ]
        root_peak = None
        if parent_window:
            max_parent_intensity = max(peak.intensity for peak in parent_window)
            intensity_threshold = max_parent_intensity * 0.1
            root_peak = min(
                (
                    peak
                    for peak in parent_window
                    if peak.intensity >= intensity_threshold
                ),
                key=lambda peak: abs(peak.mz - spectrum.precursor_mz),
            )

        fragments = [
            peak
            for peak in peaks
            if peak not in parent_window and peak.mz + 0.1 < spectrum.precursor_mz
        ]
        max_intensity = max((peak.intensity for peak in fragments), default=0.0)
        if max_intensity > 0.0 and config.min_relative_intensity > 0.0:
            fragments = [
                peak
                for peak in fragments
                if peak.intensity / max_intensity >= config.min_relative_intensity
            ]
        cap_pool = list(fragments)
        if root_peak is not None:
            cap_pool.append(root_peak)
        if len(cap_pool) > config.max_fragment_peaks:
            selected = sorted(
                cap_pool,
                key=lambda peak: peak.intensity,
                reverse=True,
            )[: config.max_fragment_peaks]
            fragments = [peak for peak in selected if peak is not root_peak]
            fragments.sort(key=lambda peak: peak.mz)
        return root_peak, fragments

    def _remove_close_lower_intensity_peaks(
        self,
        peaks: Sequence[SpectrumPeak],
        config: SiriusLikeScoringConfig,
    ) -> list[SpectrumPeak]:
        mass_sorted = sorted(peaks, key=lambda peak: peak.mz)
        deleted = [False] * len(mass_sorted)
        intensity_order = sorted(
            range(len(mass_sorted)),
            key=lambda index: (-mass_sorted[index].intensity, mass_sorted[index].mz),
        )
        for index in intensity_order:
            if deleted[index]:
                continue
            center_mz = mass_sorted[index].mz
            left = index - 1
            while left >= 0 and self._in_doubled_ms2_window(
                center_mz,
                mass_sorted[left].mz,
                config,
            ):
                deleted[left] = True
                left -= 1
            right = index + 1
            while right < len(mass_sorted) and self._in_doubled_ms2_window(
                center_mz,
                mass_sorted[right].mz,
                config,
            ):
                deleted[right] = True
                right += 1
        return [
            peak
            for index, peak in enumerate(mass_sorted)
            if not deleted[index]
        ]

    @staticmethod
    def _in_doubled_ms2_window(
        center_mz: float,
        mz: float,
        config: SiriusLikeScoringConfig,
    ) -> bool:
        window = max(
            center_mz * config.ms2_tolerance_ppm * 2e-6,
            config.mass_deviation_absolute_da * 2.0,
        )
        return abs(mz - center_mz) <= window

    def _generate_fragment_candidates(
        self,
        peaks: Sequence[SpectrumPeak],
        root_counts: Sequence[int],
        config: SiriusLikeScoringConfig,
        scorer: SiriusLikeScorer,
        root_formula: str,
        ionization: str,
        intensity_scale: float | None = None,
    ) -> dict[str, _GeneratedCandidate]:
        formula_finder = FormulaFinder(self.element_symbols, backend="rust")
        max_counts = {
            symbol: count
            for symbol, count in zip(self.element_symbols, root_counts)
            if count > 0
        }
        max_intensity = (
            max((peak.intensity for peak in peaks), default=1.0)
            if intensity_scale is None
            else intensity_scale
        )
        candidates_by_peak: list[dict[str, _GeneratedCandidate]] = []

        for peak_index, peak in enumerate(peaks, start=1):
            results = formula_finder.find_formulae(
                peak.mz,
                charge=1,
                error_ppm=config.candidate_search_ppm,
                error_da=config.candidate_search_absolute_da,
                adduct="H",
                max_counts=max_counts,
                max_results=config.candidate_limit_per_peak,
                filter_rdbe=(-1.5, 80.0),
                check_octet=False,
                backend="rust",
            )
            relative_intensity = (
                0.0 if max_intensity <= 0.0 else peak.intensity / max_intensity
            )
            peak_generated: dict[str, _GeneratedCandidate] = {}
            for result in results:
                formula = result.formula.formula
                if formula == root_formula:
                    continue
                counts = self.parse_formula_counts(formula)
                if not self.is_subformula_counts(root_counts, counts):
                    continue
                theoretical_mz = peak.mz + result.error_da
                allowed_delta = max(
                    peak.mz * config.ms2_tolerance_ppm * 1e-6,
                    config.mass_deviation_absolute_da,
                )
                if abs(peak.mz - theoretical_mz) > allowed_delta:
                    continue
                candidate_score = scorer.fragment_candidate_score(
                    formula,
                    counts,
                    peak.mz,
                    theoretical_mz,
                    result.formula.monoisotopic_mass,
                    relative_intensity,
                )
                candidate = FragmentCandidate(
                    formula=formula,
                    mass=peak.mz,
                    score=candidate_score,
                    peak_id=peak_index,
                    color=peak_index,
                    ionization=ionization,
                    intensity=peak.intensity,
                )
                item = _GeneratedCandidate(
                    candidate=candidate,
                    counts=counts,
                    theoretical_mz=theoretical_mz,
                )
                peak_generated[formula] = item
            candidates_by_peak.append(peak_generated)

        self._disjoin_nearby_fragment_candidates(peaks, candidates_by_peak, config)

        generated: dict[str, _GeneratedCandidate] = {}
        for peak_generated in candidates_by_peak:
            for formula, item in peak_generated.items():
                previous = generated.get(formula)
                if (
                    previous is None
                    or self._candidate_mass_error(item)
                    < self._candidate_mass_error(previous)
                ):
                    generated[formula] = item

        return generated

    def _disjoin_nearby_fragment_candidates(
        self,
        peaks: Sequence[SpectrumPeak],
        candidates_by_peak: Sequence[dict[str, _GeneratedCandidate]],
        config: SiriusLikeScoringConfig,
    ) -> None:
        for index in range(1, len(peaks)):
            left_peak = peaks[index - 1]
            right_peak = peaks[index]
            if not self._in_doubled_ms2_window(right_peak.mz, left_peak.mz, config):
                continue

            left_candidates = candidates_by_peak[index - 1]
            right_candidates = candidates_by_peak[index]
            for formula in set(left_candidates) & set(right_candidates):
                left_error = self._candidate_mass_error(left_candidates[formula])
                right_error = self._candidate_mass_error(right_candidates[formula])
                if left_error < right_error:
                    del right_candidates[formula]
                else:
                    del left_candidates[formula]

    @staticmethod
    def _candidate_mass_error(item: _GeneratedCandidate) -> float:
        return abs(item.candidate.mass - item.theoretical_mz)

    def _build_sirius_like_scoring(
        self,
        root: _GeneratedCandidate,
        generated: dict[str, _GeneratedCandidate],
        scorer: SiriusLikeScorer,
        intensity_scale: float | None = None,
    ) -> ExplicitFragmentationScoring:
        max_intensity = (
            max(
                (
                    item.candidate.intensity or 0.0
                    for item in generated.values()
                ),
                default=0.0,
            )
            if intensity_scale is None
            else intensity_scale
        )
        peak_scores = {
            item.candidate.color: scorer.peak_score(
                item.candidate.mass,
                0.0
                if max_intensity <= 0.0
                else (item.candidate.intensity or 0.0) / max_intensity,
            )
            for item in generated.values()
        }
        fragment_scores = {}
        for formula, item in generated.items():
            if not self.is_subformula_counts(root.counts, item.counts):
                continue
            root_loss_counts = self.subtract_counts(root.counts, item.counts)
            root_loss_formula = self.format_formula_counts(root_loss_counts)
            fragment_scores[formula] = (
                scorer.common_root_loss_score(root_loss_formula)
                + scorer.db_paired_score(formula, root.candidate.formula)
            )

        by_color = {item.candidate.color: item for item in generated.values()}
        peak_pair_scores = {}
        for child in by_color.values():
            if root.candidate.mass > child.candidate.mass:
                peak_pair_scores[(root.candidate.color, child.candidate.color)] = (
                    scorer.peak_pair_score(root.candidate.mass, child.candidate.mass)
                )
        for parent in by_color.values():
            for child in by_color.values():
                if parent.candidate.mass > child.candidate.mass:
                    peak_pair_scores[(parent.candidate.color, child.candidate.color)] = (
                        scorer.peak_pair_score(parent.candidate.mass, child.candidate.mass)
                    )

        loss_scores = {}
        for child_formula, child in generated.items():
            if root.candidate.mass <= child.candidate.mass:
                continue
            if not self.is_subformula_counts(root.counts, child.counts):
                continue
            loss_counts = self.subtract_counts(root.counts, child.counts)
            loss_formula = self.format_formula_counts(loss_counts)
            loss_scores[(root.candidate.formula, child_formula)] = scorer.loss_score(
                loss_formula,
                loss_counts,
                root.candidate.mass - child.candidate.mass,
                root.theoretical_mz - child.theoretical_mz,
                child_formula,
                True,
            )
        for parent_formula, parent in generated.items():
            for child_formula, child in generated.items():
                if parent_formula == child_formula:
                    continue
                if parent.candidate.mass <= child.candidate.mass:
                    continue
                if not self.is_subformula_counts(parent.counts, child.counts):
                    continue
                loss_counts = self.subtract_counts(parent.counts, child.counts)
                loss_formula = self.format_formula_counts(loss_counts)
                loss_scores[(parent_formula, child_formula)] = scorer.loss_score(
                    loss_formula,
                    loss_counts,
                    parent.candidate.mass - child.candidate.mass,
                    parent.theoretical_mz - child.theoretical_mz,
                    child_formula,
                )

        return ExplicitFragmentationScoring(
            peak_scores=peak_scores,
            peak_pair_scores=peak_pair_scores,
            fragment_scores=fragment_scores,
            loss_scores=loss_scores,
        )

    def _candidate_payload(
        self,
        candidates: Sequence[FragmentCandidate],
    ) -> list[tuple[str, list[int], str, int, int, float, float]]:
        return [
            (
                candidate.formula,
                list(self.parse_formula_counts(candidate.formula)),
                candidate.ionization,
                candidate.peak_id,
                candidate.tree_color,
                candidate.mass,
                candidate.score,
            )
            for candidate in candidates
        ]

    @staticmethod
    def _candidate_index(
        root_candidates: Sequence[FragmentCandidate],
        fragment_candidates: Sequence[FragmentCandidate],
    ) -> dict[str, FragmentCandidate]:
        candidates_by_formula = {}
        for candidate in [*root_candidates, *fragment_candidates]:
            if candidate.formula in candidates_by_formula:
                raise ValueError(
                    "fragmentation tree candidates must have unique formulas; "
                    f"found duplicate {candidate.formula!r}"
                )
            candidates_by_formula[candidate.formula] = candidate
        return candidates_by_formula

    @staticmethod
    def _rust_module():
        try:
            return importlib.import_module("find_mfs._rust")
        except ImportError as err:
            raise ImportError(
                "Fragmentation tree solving requires the private Rust extension. "
                "Run: uv run maturin develop --manifest-path find_mfs/rust/Cargo.toml"
            ) from err

    @classmethod
    def _normalize_element_symbols(cls, elements: Iterable[str] | str) -> list[str]:
        if isinstance(elements, str):
            return list(cls._rust_module().parse_element_symbols(elements))
        return list(elements)
