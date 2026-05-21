from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .results import SpectrumPeak


@dataclass(frozen=True, slots=True)
class FragmentationSpectrum:
    """Raw MS/MS spectrum input for default fragmentation-tree scoring."""

    precursor_mz: float
    peaks: list[SpectrumPeak]
    precursor_formula: str | None = None
    precursor_ion: str = "[M+H]+"
    name: str | None = None
    accession: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
