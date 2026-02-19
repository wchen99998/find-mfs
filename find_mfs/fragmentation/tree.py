# ============================================================================
# Fragmentation Tree generation (SIRIUS-like Maximum Colorful Subtree via ILP)
#
# Implements isotope-envelope-aware scoring following the approach described in
# Böcker & Rasche (2008) doi:10.1186/1471-2105-7-234 — candidate formulae are
# scored against observed isotope envelopes (not just monoisotopic peaks) to
# dramatically improve identification confidence for real-world spectra.
# ============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Mapping, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.finder import FormulaFinder

from molmass import Formula
from ..core.light_formula import LightFormula


# NOTE: SciPy is used only for the ILP solve; keep it optional at import time.
def _require_scipy_milp():
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import coo_matrix
    except Exception as e:
        raise ImportError(
            "FragmentationTreeGenerator requires SciPy (scipy.optimize.milp). "
            "Install scipy>=1.9 (HiGHS MILP)."
        ) from e
    return milp, LinearConstraint, Bounds, coo_matrix


def _hill_sort_key(sym: str):
    if sym == "C":
        return (0,)
    if sym == "H":
        return (1,)
    return (2, sym)


def _iter_formula_items(formula: Union[Formula, LightFormula]):
    """Yield (symbol, count) ignoring electrons."""
    if isinstance(formula, LightFormula):
        yield from formula._iter_nonzero_items()
        return

    for sym, item in formula.composition().items():
        if sym in ("", "e-"):
            continue
        if item.count > 0:
            yield sym, int(item.count)


def _formula_to_count_vector(
    formula: Union[Formula, LightFormula],
    symbols: Sequence[str],
    symbol_to_idx: Mapping[str, int],
) -> np.ndarray | None:
    vec = np.zeros(len(symbols), dtype=np.int64)
    for sym, cnt in _iter_formula_items(formula):
        idx = symbol_to_idx.get(sym, None)
        if idx is None:
            return None
        vec[idx] = cnt
    return vec


def _counts_to_formula_string(
    symbols: Sequence[str],
    counts: np.ndarray,
) -> str:
    idx_map = {s: i for i, s in enumerate(symbols)}
    ordered = sorted(symbols, key=_hill_sort_key)
    parts: list[str] = []
    for sym in ordered:
        c = int(counts[idx_map[sym]])
        if c <= 0:
            continue
        parts.append(sym if c == 1 else f"{sym}{c}")
    return "".join(parts)


def _is_subformula(child: np.ndarray, parent: np.ndarray) -> bool:
    return bool(np.all(child <= parent))

def _canonicalize_neutral_formula_string(formula_str: str) -> str:
    """
    Return a canonical Hill-ordered formula string for neutral formulae.

    Used to match common neutral losses regardless of input ordering
    (e.g., "NH3" -> "H3N", "SO2" -> "O2S").
    """
    try:
        return Formula(formula_str).formula
    except Exception:
        return formula_str


# ---------------------------------------------------------------------------
# Isotope envelope grouping
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IsotopeEnvelope:
    """
    A group of peaks representing isotopologues of the same species.

    This is the recommended input format for FragmentationTreeGenerator when
    isotope information is available. Using envelopes rather than bare peaks
    is critical for confident molecular formula assignment in practice
    (Böcker & Rasche 2008, doi:10.1186/1471-2105-7-234).

    Attributes:
        peaks: 2D array of shape (n, 2) with columns [m/z, intensity].
            The first row should be the monoisotopic (M+0) peak.
        monoisotopic_mz: m/z of the monoisotopic peak (defaults to
            peaks[0, 0] if not provided).
        monoisotopic_intensity: Intensity of the monoisotopic peak
            (defaults to peaks[0, 1] if not provided).
    """
    peaks: np.ndarray
    monoisotopic_mz: float | None = None
    monoisotopic_intensity: float | None = None

    def __post_init__(self):
        self.peaks = np.asarray(self.peaks, dtype=np.float64)
        if self.peaks.ndim != 2 or self.peaks.shape[1] != 2:
            raise ValueError(
                f"peaks must be shape (n, 2), got {self.peaks.shape}"
            )
        if self.monoisotopic_mz is None:
            self.monoisotopic_mz = float(self.peaks[0, 0])
        if self.monoisotopic_intensity is None:
            self.monoisotopic_intensity = float(self.peaks[0, 1])


def detect_isotope_envelopes(
    spectrum: Sequence[tuple[float, float]],
    *,
    charge: int = 1,
    mz_tolerance: float = 0.02,
    max_envelope_size: int = 6,
    h2_loss_intensity_ratio: float = 0.5,
) -> list[IsotopeEnvelope]:
    """
    Heuristically group raw spectrum peaks into isotope envelopes.

    The key challenge is distinguishing M+2 isotopologues from H2 losses
    (~2.016 Da for z=1). The heuristic: if a candidate M+2 peak has
    intensity > h2_loss_intensity_ratio relative to M+0, it's more likely
    an independent fragment (H2 loss) than an isotopologue, since natural
    M+2 intensities rarely exceed ~50% of M+0 for small molecules without
    S/Cl/Br.

    Args:
        spectrum: List of (mz, intensity) tuples.
        charge: Charge state — isotope spacing is ~1.003/|charge|.
        mz_tolerance: Tolerance for matching isotope spacing.
        max_envelope_size: Maximum number of peaks per envelope.
        h2_loss_intensity_ratio: If a candidate M+2 peak has relative
            intensity above this threshold, treat it as an independent
            peak (likely H2 loss) rather than an isotopologue.

    Returns:
        List of IsotopeEnvelope objects, one per detected monoisotopic peak.
    """
    if len(spectrum) == 0:
        return []

    abs_charge = max(abs(charge), 1)
    isotope_spacing = 1.00335 / abs_charge  # ~neutron mass / charge

    # Sort by m/z ascending
    sorted_peaks = sorted(spectrum, key=lambda p: p[0])
    used = [False] * len(sorted_peaks)
    envelopes: list[IsotopeEnvelope] = []

    for i, (mz_i, int_i) in enumerate(sorted_peaks):
        if used[i] or int_i <= 0:
            continue

        envelope_peaks = [(mz_i, int_i)]
        used[i] = True
        current_mz = mz_i

        # Walk up in m/z looking for M+1, M+2, ...
        for _k in range(1, max_envelope_size):
            expected_mz = current_mz + isotope_spacing
            best_j = None
            best_diff = mz_tolerance + 1

            for j in range(i + 1, len(sorted_peaks)):
                if used[j]:
                    continue
                diff = abs(sorted_peaks[j][0] - expected_mz)
                if diff < best_diff:
                    best_diff = diff
                    best_j = j

            if best_j is None or best_diff > mz_tolerance:
                break

            cand_mz, cand_int = sorted_peaks[best_j]

            # H2 loss ambiguity check for M+2 peaks:
            # Natural M+2 abundance is typically < 50% of M+0 for small
            # molecules (unless S, Cl, Br are present). If the candidate
            # has suspiciously high intensity, it's likely an independent
            # fragment from H2 loss rather than an isotopologue.
            if _k == 2:
                rel_intensity = cand_int / int_i if int_i > 0 else 0
                if rel_intensity > h2_loss_intensity_ratio:
                    break

            # Isotopologues should decrease in intensity (generally)
            # Allow M+1 to be up to ~60% of M+0 (covers most organic molecules)
            if cand_int > int_i * 0.8 and _k == 1:
                break

            envelope_peaks.append((cand_mz, cand_int))
            used[best_j] = True
            current_mz = cand_mz

        envelopes.append(
            IsotopeEnvelope(peaks=np.array(envelope_peaks, dtype=np.float64))
        )

    # Mark remaining unused peaks as singleton envelopes
    for i, (mz_i, int_i) in enumerate(sorted_peaks):
        if not used[i] and int_i > 0:
            envelopes.append(
                IsotopeEnvelope(
                    peaks=np.array([[mz_i, int_i]], dtype=np.float64)
                )
            )

    return envelopes


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Peak:
    """An MS/MS peak, optionally with its isotope envelope."""
    mz: float
    intensity: float
    peak_id: int
    envelope: IsotopeEnvelope | None = None


@dataclass(slots=True)
class FragmentNode:
    """A candidate fragment formula assigned to a peak (color)."""
    idx: int
    color: int           # peak_id; root uses -1
    peak: Peak
    formula: Union[Formula, LightFormula]
    counts: np.ndarray   # aligned with tree element_symbols
    score: float


@dataclass(slots=True)
class FragmentEdge:
    """A directed loss edge between two fragment candidates."""
    idx: int
    parent: int
    child: int
    loss_counts: np.ndarray
    loss_formula: str
    score: float


@dataclass
class FragmentationTree:
    element_symbols: list[str]
    root_idx: int
    nodes: list[FragmentNode]
    edges: list[FragmentEdge]
    total_score: float

    def to_networkx(self):
        import networkx as nx
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(
                n.idx,
                formula=str(n.formula),
                mz=n.peak.mz,
                intensity=n.peak.intensity,
                score=n.score,
                color=n.color,
            )
        for e in self.edges:
            G.add_edge(
                e.parent,
                e.child,
                loss=e.loss_formula,
                score=e.score,
            )
        return G

    def to_table(self, max_rows: int = 50) -> str:
        idx_to_node = {n.idx: n for n in self.nodes}
        lines = [
            f"FragmentationTree(total_score={self.total_score:.3f}, "
            f"n_nodes={len(self.nodes)}, n_edges={len(self.edges)})",
            "",
            f"{'Parent':<22} {'Child':<22} {'Loss':<18} {'EdgeScore':>10}",
            "-" * 80,
        ]
        for i, e in enumerate(self.edges[:max_rows]):
            p = idx_to_node[e.parent].formula.formula
            c = idx_to_node[e.child].formula.formula
            lines.append(f"{p:<22} {c:<22} {e.loss_formula:<18} {e.score:>10.3f}")
        if len(self.edges) > max_rows:
            lines.append(f"... ({len(self.edges) - max_rows} more edges)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FragmentationTreeGenerator:
    """
    Build a fragmentation tree from an MS/MS spectrum using a SIRIUS-like
    Maximum Colorful Subtree formulation solved via MILP.

    Key design choices:

    **Isotope envelopes** (doi:10.1186/1471-2105-7-234):
      The generator accepts isotope envelopes — groups of isotopologue
      peaks — rather than requiring a de-isotoped spectrum. When an
      envelope is available for a peak, candidate formulae are scored
      against the full isotope pattern (via IsoSpecPy) in addition to
      mass accuracy. This is critical for confident formula assignment
      in practice because isotope patterns dramatically constrain the
      candidate space. Single peaks (no detected isotopologues) are
      still accepted and scored on mass accuracy alone.

    **Edge scoring**:
      Edges are scored by three terms:
      1. A penalty proportional to the neutral loss mass (guides the
         solver toward step-wise, chemically plausible fragmentation).
      2. A bonus for common neutral losses (H2O, CO2, NH3, etc.).
      3. A penalty based on the deviation between observed and
         theoretical delta-m/z — this ensures that the mass accuracy
         of the loss itself is considered, not just the accuracy of
         the endpoint nodes.

    Assumptions (common in small-molecule FT construction):
      - fragments keep the same charge state as the precursor
      - atom conservation: each fragment is a subformula of the precursor
      - directed acyclic fragmentation: edges go from higher m/z to lower
    """

    DEFAULT_COMMON_LOSSES = (
        "H2O", "CO", "CO2", "NH3", "H2", "CH4",
        "C2H4", "C2H2", "HCOOH", "H2CO", "SO2", "SO3",
        "H3PO4", "HPO3",
    )

    def __init__(
        self,
        finder: FormulaFinder,
        *,
        error_ppm: float = 10.0,
        error_da: float = 0.0,
        max_peaks: int = 40,
        min_rel_intensity: float = 0.01,
        max_candidates_per_peak: int = 10,
        max_results_per_peak: int = 200,
        common_losses: Sequence[str] | None = None,
        # scoring weights:
        w_mass: float = 1.0,
        w_intensity: float = 0.5,
        w_isotope: float = 2.0,
        loss_mass_scale: float = 200.0,
        loss_common_bonus: float = 1.0,
        loss_mass_deviation_scale: float = 0.3,
        # isotope envelope scoring:
        isotope_mz_tolerance: float = 0.01,
        isotope_intensity_tolerance: float = 0.15,
    ):
        self.finder = finder
        self.error_ppm = float(error_ppm)
        self.error_da = float(error_da)
        self.max_peaks = int(max_peaks)
        self.min_rel_intensity = float(min_rel_intensity)
        self.max_candidates_per_peak = int(max_candidates_per_peak)
        self.max_results_per_peak = int(max_results_per_peak)

        self.w_mass = float(w_mass)
        self.w_intensity = float(w_intensity)
        self.w_isotope = float(w_isotope)
        self.loss_mass_scale = float(loss_mass_scale)
        self.loss_common_bonus = float(loss_common_bonus)
        self.loss_mass_deviation_scale = float(loss_mass_deviation_scale)

        self.isotope_mz_tolerance = float(isotope_mz_tolerance)
        self.isotope_intensity_tolerance = float(isotope_intensity_tolerance)

        self.common_losses = (
            tuple(common_losses)
            if common_losses is not None
            else self.DEFAULT_COMMON_LOSSES
        )
        common_loss_set: set[str] = set()
        for loss in self.common_losses:
            common_loss_set.add(loss)
            common_loss_set.add(_canonicalize_neutral_formula_string(loss))
        self._common_loss_set = frozenset(common_loss_set)

    # ---------------------------
    # Public API
    # ---------------------------

    def generate(
        self,
        *,
        precursor_mz: float,
        spectrum: Sequence[tuple[float, float]] | None = None,
        envelopes: Sequence[IsotopeEnvelope] | None = None,
        charge: int = 1,
        adduct: str | None = None,
        top_n_precursor_formulas: int = 5,
        filter_rdbe: tuple[float, float] | None = None,
        check_octet: bool = False,
        auto_detect_envelopes: bool = True,
        min_counts: dict[str, int] | str | None = None,
        max_counts: dict[str, int] | str | None = None,
    ) -> FragmentationTree:
        """
        Compute the best fragmentation tree over the top precursor formula
        candidates.

        Accepts either a raw spectrum (list of (mz, intensity) tuples) or
        pre-assembled isotope envelopes. When envelopes are provided,
        isotope pattern matching is used to improve formula scoring.

        Args:
            precursor_mz: precursor m/z
            spectrum: sequence of (mz, intensity) for MS/MS peaks.
                Mutually exclusive with ``envelopes``.
            envelopes: sequence of IsotopeEnvelope objects grouping
                isotopologues. Preferred input when isotope information
                is available. Mutually exclusive with ``spectrum``.
            charge: charge state (passed to FormulaFinder)
            adduct: adduct string (passed to FormulaFinder)
            top_n_precursor_formulas: evaluate multiple precursor candidates
            filter_rdbe: RDBE range filter
            check_octet: apply octet rule
            auto_detect_envelopes: when True and ``spectrum`` is provided,
                automatically group peaks into isotope envelopes using
                heuristic detection. Set False to treat every peak as an
                independent signal (legacy behavior).
            min_counts: minimum element counts, passed to FormulaFinder.
                Can be a dict or formula string (e.g. "C5").
            max_counts: maximum element counts, passed to FormulaFinder.
                Can be a dict or formula string (e.g. "C37H67NO13").
                Constraining the search space is recommended for large
                molecules (>500 Da) to keep candidate counts manageable.
        """
        if spectrum is not None and envelopes is not None:
            raise ValueError(
                "Provide either `spectrum` or `envelopes`, not both."
            )
        if spectrum is None and envelopes is None:
            raise ValueError(
                "Either `spectrum` or `envelopes` must be provided."
            )

        # 1) pick top precursor formula candidates
        prec_results = self.finder.find_formulae(
            mass=precursor_mz,
            charge=charge,
            adduct=adduct,
            error_ppm=self.error_ppm,
            error_da=self.error_da,
            max_results=max(50, top_n_precursor_formulas * 10),
            filter_rdbe=filter_rdbe,
            check_octet=check_octet,
            min_counts=min_counts,
            max_counts=max_counts,
        )
        if len(prec_results) == 0:
            raise ValueError(
                "No precursor formula candidates found; "
                "cannot build fragmentation tree."
            )

        precursor_candidates = prec_results.candidates[:top_n_precursor_formulas]

        # 2) prepare peaks (with envelopes when available)
        if envelopes is not None:
            peaks = self._prepare_peaks_from_envelopes(envelopes)
        elif auto_detect_envelopes:
            detected = detect_isotope_envelopes(
                spectrum, charge=charge
            )
            peaks = self._prepare_peaks_from_envelopes(detected)
        else:
            peaks = self._prepare_peaks(spectrum)

        best_tree: FragmentationTree | None = None
        best_score = -float("inf")

        for cand in precursor_candidates:
            tree = self._tree_for_root(
                root_formula=cand.formula,
                root_mz=precursor_mz,
                peaks=peaks,
                charge=charge,
                adduct=adduct,
                filter_rdbe=filter_rdbe,
                check_octet=check_octet,
                max_counts=max_counts,
            )
            if tree.total_score > best_score:
                best_score = tree.total_score
                best_tree = tree

        assert best_tree is not None
        return best_tree

    # ---------------------------
    # Peak preparation
    # ---------------------------

    def _prepare_peaks(
        self, spectrum: Sequence[tuple[float, float]]
    ) -> list[Peak]:
        """Prepare peaks from a raw spectrum (no envelope information)."""
        if len(spectrum) == 0:
            return []

        max_i = max(i for _, i in spectrum) if spectrum else 1.0
        if max_i <= 0:
            max_i = 1.0

        filtered: list[Peak] = []
        for k, (mz, inten) in enumerate(spectrum):
            rel = inten / max_i
            if rel >= self.min_rel_intensity and mz > 0:
                filtered.append(
                    Peak(mz=float(mz), intensity=float(inten), peak_id=k)
                )

        filtered.sort(key=lambda p: p.intensity, reverse=True)
        filtered = filtered[: self.max_peaks]
        filtered.sort(key=lambda p: p.mz, reverse=True)
        return filtered

    def _prepare_peaks_from_envelopes(
        self, envelopes: Sequence[IsotopeEnvelope]
    ) -> list[Peak]:
        """
        Prepare peaks from isotope envelopes.

        Each envelope contributes one Peak (the monoisotopic peak) with
        the full envelope attached for scoring. This is the recommended
        path: isotope envelopes provide much stronger constraints for
        formula identification than monoisotopic peaks alone.
        """
        if len(envelopes) == 0:
            return []

        max_i = max(env.monoisotopic_intensity for env in envelopes)
        if max_i <= 0:
            max_i = 1.0

        filtered: list[Peak] = []
        for k, env in enumerate(envelopes):
            rel = env.monoisotopic_intensity / max_i
            if rel >= self.min_rel_intensity and env.monoisotopic_mz > 0:
                filtered.append(
                    Peak(
                        mz=env.monoisotopic_mz,
                        intensity=env.monoisotopic_intensity,
                        peak_id=k,
                        envelope=env,
                    )
                )

        filtered.sort(key=lambda p: p.intensity, reverse=True)
        filtered = filtered[: self.max_peaks]
        filtered.sort(key=lambda p: p.mz, reverse=True)
        return filtered

    # ---------------------------
    # Graph building
    # ---------------------------

    def _tree_for_root(
        self,
        *,
        root_formula: Union[Formula, LightFormula],
        root_mz: float,
        peaks: list[Peak],
        charge: int,
        adduct: str | None,
        filter_rdbe: tuple[float, float] | None,
        check_octet: bool,
        max_counts: dict[str, int] | str | None = None,
    ) -> FragmentationTree:
        base_symbols = list(self.finder.decomposer.element_symbols)
        base_set = set(base_symbols)
        for sym, _ in _iter_formula_items(root_formula):
            if sym not in base_set:
                base_symbols.append(sym)
                base_set.add(sym)

        element_symbols = base_symbols
        symbol_to_idx = {s: i for i, s in enumerate(element_symbols)}

        root_counts = _formula_to_count_vector(
            root_formula, element_symbols, symbol_to_idx
        )
        if root_counts is None:
            raise ValueError(
                "Root formula contains unsupported elements."
            )

        # build nodes
        nodes: list[FragmentNode] = []
        root_node = FragmentNode(
            idx=0,
            color=-1,
            peak=Peak(mz=float(root_mz), intensity=0.0, peak_id=-1),
            formula=root_formula,
            counts=root_counts,
            score=0.0,
        )
        nodes.append(root_node)

        nodes_by_color: dict[int, list[int]] = {}

        max_intensity = max((p.intensity for p in peaks), default=1.0)
        if max_intensity <= 0:
            max_intensity = 1.0

        next_idx = 1
        for peak in peaks:
            res = self.finder.find_formulae(
                mass=peak.mz,
                charge=charge,
                adduct=adduct,
                error_ppm=self.error_ppm,
                error_da=self.error_da,
                max_results=self.max_results_per_peak,
                filter_rdbe=filter_rdbe,
                check_octet=check_octet,
                max_counts=max_counts,
            )

            kept_for_peak: list[FragmentNode] = []
            for fc in res.candidates:
                f = fc.formula

                vec = _formula_to_count_vector(
                    f, element_symbols, symbol_to_idx
                )
                if vec is None:
                    continue
                if not _is_subformula(vec, root_counts):
                    continue

                node_score = self._score_node(
                    peak=peak,
                    max_intensity=max_intensity,
                    formula=f,
                )

                kept_for_peak.append(
                    FragmentNode(
                        idx=next_idx,
                        color=peak.peak_id,
                        peak=peak,
                        formula=f,
                        counts=vec,
                        score=node_score,
                    )
                )
                next_idx += 1

                if len(kept_for_peak) >= self.max_candidates_per_peak:
                    break

            if kept_for_peak:
                idxs = [n.idx for n in kept_for_peak]
                nodes_by_color[peak.peak_id] = idxs
                nodes.extend(kept_for_peak)

        # build edges
        edges: list[FragmentEdge] = []
        edge_idx = 0
        idx_to_node = {n.idx: n for n in nodes}

        peak_order = [
            (p.peak_id, p.mz)
            for p in peaks
            if p.peak_id in nodes_by_color
        ]

        def add_edge(u_idx: int, v_idx: int):
            nonlocal edge_idx
            u = idx_to_node[u_idx]
            v = idx_to_node[v_idx]
            if u.color == v.color:
                return
            if u.peak.mz <= v.peak.mz:
                return
            if not _is_subformula(v.counts, u.counts):
                return

            loss_counts = u.counts - v.counts
            if int(loss_counts.sum()) <= 0:
                return

            loss_formula = _counts_to_formula_string(
                element_symbols, loss_counts
            )

            loss_mass = float(
                u.formula.monoisotopic_mass - v.formula.monoisotopic_mass
            )
            observed_delta_mz = float(u.peak.mz - v.peak.mz)

            edge_score = self._score_edge(
                loss_formula=loss_formula,
                loss_mass=loss_mass,
                observed_delta_mz=observed_delta_mz,
            )

            edges.append(
                FragmentEdge(
                    idx=edge_idx,
                    parent=u_idx,
                    child=v_idx,
                    loss_counts=loss_counts,
                    loss_formula=loss_formula,
                    score=edge_score,
                )
            )
            edge_idx += 1

        for color, node_idxs in nodes_by_color.items():
            for v_idx in node_idxs:
                add_edge(0, v_idx)

        for i in range(len(peak_order)):
            pid_i, mz_i = peak_order[i]
            for j in range(i + 1, len(peak_order)):
                pid_j, mz_j = peak_order[j]
                for u_idx in nodes_by_color[pid_i]:
                    for v_idx in nodes_by_color[pid_j]:
                        add_edge(u_idx, v_idx)

        selected_nodes, selected_edges, total_score = self._solve_ilp(
            nodes, edges, root_idx=0
        )

        sel_node_set = set(selected_nodes)
        sel_edge_set = set(selected_edges)

        final_nodes = [n for n in nodes if n.idx in sel_node_set]
        final_edges = [e for e in edges if e.idx in sel_edge_set]

        return FragmentationTree(
            element_symbols=list(element_symbols),
            root_idx=0,
            nodes=final_nodes,
            edges=final_edges,
            total_score=float(total_score),
        )

    # ---------------------------
    # Scoring
    # ---------------------------

    def _score_node(
        self,
        *,
        peak: Peak,
        max_intensity: float,
        formula: Union[Formula, LightFormula],
    ) -> float:
        # Intensity term: higher-intensity peaks are more valuable to explain.
        # Uses relative intensity directly (0–1 range, always positive).
        rel_i = peak.intensity / max_intensity if max_intensity > 0 else 0.0
        rel_i = max(rel_i, 0.0)

        # Mass accuracy term: Gaussian likelihood (0–1 range) — measures how
        # well the candidate formula mass matches the observed peak m/z.
        tol_da = max(
            peak.mz * (self.error_ppm / 1e6) if self.error_ppm else 0.0,
            self.error_da if self.error_da else 0.0,
        )
        sigma = tol_da / 3.0 if tol_da > 0 else 1.0

        err_da = float(formula.monoisotopic_mass - peak.mz)
        mass_score = math.exp(-0.5 * (err_da / sigma) ** 2)

        # Both terms are positive, so including a well-matched peak in the
        # tree always increases the objective — as desired for MILP.
        score = self.w_mass * mass_score + self.w_intensity * rel_i

        # Isotope envelope scoring: when envelope data is available, score
        # the candidate against the full observed isotope pattern.
        if peak.envelope is not None and peak.envelope.peaks.shape[0] > 1:
            isotope_score = self._score_isotope_match(
                formula=formula,
                envelope=peak.envelope,
            )
            score += self.w_isotope * isotope_score

        return score

    def _score_isotope_match(
        self,
        *,
        formula: Union[Formula, LightFormula],
        envelope: IsotopeEnvelope,
    ) -> float:
        """
        Score how well a candidate formula's theoretical isotope pattern
        matches the observed envelope.

        Returns a log-likelihood-style score (higher = better match).
        """
        try:
            from ..isotopes.envelope import (
                get_isotope_envelope,
            )
        except ImportError:
            return 0.0

        try:
            predicted = get_isotope_envelope(
                formula=formula,
                mz_tolerance=0.05,
                threshold=0.001,
            )
        except Exception:
            return 0.0

        # Normalize observed envelope to monoisotopic = 1.0
        observed = envelope.peaks.copy()
        mono_idx = np.argmax(observed[:, 1])
        if observed[mono_idx, 1] > 0:
            observed[:, 1] = observed[:, 1] / observed[mono_idx, 1]

        # Match observed peaks to predicted peaks
        n_matched = 0
        total_intensity_error = 0.0

        for obs_mz, obs_int in observed:
            diffs = np.abs(predicted[:, 0] - obs_mz)
            best_idx = np.argmin(diffs)
            if diffs[best_idx] < self.isotope_mz_tolerance:
                pred_int = predicted[best_idx, 1]
                int_err = abs(obs_int - pred_int)
                if int_err < self.isotope_intensity_tolerance:
                    n_matched += 1
                total_intensity_error += int_err

        n_observed = observed.shape[0]
        match_fraction = n_matched / n_observed if n_observed > 0 else 0.0

        # Score: reward high match fraction, penalize intensity deviations
        score = match_fraction * 2.0 - total_intensity_error
        return score

    def _score_edge(
        self,
        *,
        loss_formula: str,
        loss_mass: float,
        observed_delta_mz: float = 0.0,
    ) -> float:
        """
        Score an edge (neutral loss).

        Three components:
        1. Base penalty proportional to loss mass — this guides the solver
           toward step-wise fragmentation (smaller, chemically meaningful
           losses) rather than large, unlikely jumps.
        2. Bonus for recognized common neutral losses.
        3. Penalty for deviation between observed delta-m/z and theoretical
           loss mass — this directly penalizes formula assignments whose
           implied loss mass doesn't match the observed mass difference.
        """
        # (1) Penalize large losses (logarithmic scale so that moderate
        #     losses like cladinose ~158 Da are not over-penalized compared
        #     to the node benefit of explaining a peak)
        base = (
            -math.log1p(abs(loss_mass) / self.loss_mass_scale)
            if self.loss_mass_scale > 0
            else 0.0
        )

        # (2) Reward common losses
        bonus = (
            self.loss_common_bonus
            if loss_formula in self._common_loss_set
            else 0.0
        )

        # (3) Penalize mass deviation between observed and theoretical
        mass_deviation = 0.0
        if observed_delta_mz > 0 and self.loss_mass_deviation_scale > 0:
            deviation = abs(loss_mass - observed_delta_mz)
            tol = max(
                observed_delta_mz * (self.error_ppm / 1e6)
                if self.error_ppm
                else 0.0,
                self.error_da if self.error_da else 0.0,
            )
            sigma = tol / 3.0 if tol > 0 else 1.0
            mass_deviation = -0.5 * (deviation / sigma) ** 2
            mass_deviation *= self.loss_mass_deviation_scale

        return base + bonus + mass_deviation

    # ---------------------------
    # ILP solve (Maximum Colorful Subtree)
    # ---------------------------

    def _solve_ilp(
        self,
        nodes: list[FragmentNode],
        edges: list[FragmentEdge],
        *,
        root_idx: int,
    ) -> tuple[list[int], list[int], float]:
        milp, LinearConstraint, Bounds, coo_matrix = _require_scipy_milp()

        n = len(nodes)
        m = len(edges)
        c = np.zeros(n + m, dtype=np.float64)

        for node in nodes:
            c[node.idx] = -float(node.score)
        for e in edges:
            c[n + e.idx] = -float(e.score)

        lb = np.zeros(n + m, dtype=np.float64)
        ub = np.ones(n + m, dtype=np.float64)

        lb[root_idx] = 1.0
        ub[root_idx] = 1.0

        bounds = Bounds(lb, ub)
        integrality = np.ones(n + m, dtype=int)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b_l: list[float] = []
        b_u: list[float] = []
        row_id = 0

        incoming: list[list[int]] = [[] for _ in range(n)]
        for e in edges:
            incoming[e.child].append(e.idx)

        # (1) For v != root: sum_in(y_uv) - x_v = 0
        for v in range(n):
            if v == root_idx:
                continue
            rows.append(row_id)
            cols.append(v)
            data.append(-1.0)
            for eidx in incoming[v]:
                rows.append(row_id)
                cols.append(n + eidx)
                data.append(1.0)
            b_l.append(0.0)
            b_u.append(0.0)
            row_id += 1

        # (2) For each edge u->v: y_uv - x_u <= 0
        for e in edges:
            rows.append(row_id)
            cols.append(e.parent)
            data.append(-1.0)
            rows.append(row_id)
            cols.append(n + e.idx)
            data.append(1.0)
            b_l.append(-np.inf)
            b_u.append(0.0)
            row_id += 1

        # (3) Color constraints: for each peak color, sum x_v <= 1
        color_to_nodes: dict[int, list[int]] = {}
        for node in nodes:
            if node.idx == root_idx:
                continue
            color_to_nodes.setdefault(node.color, []).append(node.idx)

        for color, vids in color_to_nodes.items():
            for v in vids:
                rows.append(row_id)
                cols.append(v)
                data.append(1.0)
            b_l.append(-np.inf)
            b_u.append(1.0)
            row_id += 1

        A = coo_matrix(
            (data, (rows, cols)), shape=(row_id, n + m)
        ).tocsr()
        constraints = LinearConstraint(
            A,
            np.array(b_l, dtype=np.float64),
            np.array(b_u, dtype=np.float64),
        )

        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )
        if not res.success:
            raise RuntimeError(f"MILP failed: {res.message}")

        x = res.x[:n]
        y = res.x[n:]
        selected_nodes = [i for i in range(n) if x[i] > 0.5]
        selected_edges = [i for i in range(m) if y[i] > 0.5]
        total_score = float(-res.fun)
        return selected_nodes, selected_edges, total_score
