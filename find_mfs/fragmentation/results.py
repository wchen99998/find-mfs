from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class SpectrumPeak:
    """A measured MS/MS peak used for fragmentation-tree annotation."""

    mz: float
    intensity: float
    peak_id: int | None = None


@dataclass(frozen=True, slots=True)
class FragmentCandidate:
    """A candidate formula assignment for one spectrum peak."""

    formula: str
    mass: float
    score: float = 0.0
    peak_id: int = 0
    color: int | None = None
    ionization: str = "[M+H]+"
    intensity: float | None = None

    @property
    def tree_color(self) -> int:
        return self.peak_id if self.color is None else self.color


@dataclass(frozen=True, slots=True)
class ExplicitFragmentationScoring:
    """
    Explicit SIRIUS-style graph scoring terms.

    Fragment candidate scores live on ``FragmentCandidate.score``. The maps
    below add peak, peak-pair, formula, loss, and graph-level terms before the
    Rust ILP solver selects the optimal colorful subtree.
    """

    peak_scores: dict[int, float] = field(default_factory=dict)
    peak_pair_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    fragment_scores: dict[str, float] = field(default_factory=dict)
    loss_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    general_graph_score: float = 0.0


@dataclass(frozen=True, slots=True)
class FragmentationTreeOptions:
    """Solver and graph-processing options for fragmentation tree search."""

    reduce_graph: bool = True
    minimal_score: float | None = None
    time_limit_seconds: float | None = None
    threads: int | None = None


@dataclass(frozen=True, slots=True)
class Fragment:
    """A selected fragment in a fragmentation tree."""

    formula: str
    counts: tuple[int, ...]
    ionization: str
    peak_id: int | None
    color: int
    mass: float
    candidate_score: float
    intensity: float | None = None


@dataclass(frozen=True, slots=True)
class Loss:
    """A selected fragmentation event between two selected fragments."""

    source: Fragment
    target: Fragment
    formula: str
    score: float


@dataclass(frozen=True, slots=True)
class FragmentationTree:
    """The optimal selected colorful fragmentation tree."""

    root: Fragment
    fragments: list[Fragment]
    losses: list[Loss]
    tree_score: float
    is_optimal: bool
    solver_status: str
    graph_vertex_count: int
    graph_edge_count: int
    reduced_vertex_count: int
    reduced_edge_count: int
    query_params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            "FragmentationTree("
            f"root={self.root.formula!r}, "
            f"fragments={len(self.fragments)}, "
            f"losses={len(self.losses)}, "
            f"tree_score={self.tree_score:.6g}, "
            f"status={self.solver_status!r})"
        )

    def to_table(self, max_rows: int | None = None) -> str:
        rows = self.fragments if max_rows is None else self.fragments[:max_rows]
        if not rows:
            return "No fragments selected."

        header = (
            f"{'Formula':<18} {'Peak':>6} {'Color':>6} "
            f"{'Mass':>12} {'Score':>12} {'Intensity':>12}"
        )
        sep = "-" * len(header)
        body = [
            f"{fragment.formula:<18} "
            f"{self._optional_value(fragment.peak_id):>6} "
            f"{fragment.color:>6} "
            f"{fragment.mass:>12.6f} "
            f"{fragment.candidate_score:>12.6f} "
            f"{self._optional_value(fragment.intensity):>12}"
            for fragment in rows
        ]
        lines = [header, sep] + body
        if max_rows is not None and len(self.fragments) > max_rows:
            lines.append(f"... and {len(self.fragments) - max_rows} more")
        return "\n".join(lines)

    def losses_table(self) -> str:
        if not self.losses:
            return "No losses selected."

        header = f"{'Source':<18} {'Target':<18} {'Loss':<14} {'Score':>12}"
        sep = "-" * len(header)
        body = [
            f"{loss.source.formula:<18} {loss.target.formula:<18} "
            f"{loss.formula:<14} {loss.score:>12.6f}"
            for loss in self.losses
        ]
        return "\n".join([header, sep] + body)

    def to_dataframe(self) -> "pd.DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame(
            [
                {
                    "formula": fragment.formula,
                    "peak_id": fragment.peak_id,
                    "color": fragment.color,
                    "mass": fragment.mass,
                    "candidate_score": fragment.candidate_score,
                    "intensity": fragment.intensity,
                }
                for fragment in self.fragments
            ]
        )

    def losses_dataframe(self) -> "pd.DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for losses_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame(
            [
                {
                    "source": loss.source.formula,
                    "target": loss.target.formula,
                    "loss": loss.formula,
                    "score": loss.score,
                }
                for loss in self.losses
            ]
        )

    @staticmethod
    def _optional_value(value: int | float | None) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)


@dataclass(frozen=True, slots=True)
class FragmentationTreeSearchResults:
    """Container for multiple precursor-candidate fragmentation trees."""

    trees: list[FragmentationTree]
    query_params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trees)

    def __iter__(self) -> Iterable[FragmentationTree]:
        return iter(self.trees)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return FragmentationTreeSearchResults(
                self.trees[idx],
                query_params=self.query_params,
            )
        return self.trees[idx]

    def __repr__(self) -> str:
        return f"FragmentationTreeSearchResults(n_trees={len(self.trees)})"
