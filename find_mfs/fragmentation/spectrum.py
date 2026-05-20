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

    @classmethod
    def from_massbank_record(cls, record: dict[str, Any]) -> "FragmentationSpectrum":
        focused_ion = record.get("mass_spectrometry", {}).get("focused_ion", [])
        focused = {
            item.get("subtag"): item.get("value")
            for item in focused_ion
            if isinstance(item, dict)
        }
        precursor_mz = float(focused["PRECURSOR_M/Z"])
        precursor_ion = str(focused.get("PRECURSOR_TYPE", "[M+H]+"))
        compound = record.get("compound", {})
        peak_values = record.get("peak", {}).get("peak", {}).get("values", [])
        peaks = [
            SpectrumPeak(
                mz=float(peak["mz"]),
                intensity=float(peak.get("intensity", peak.get("rel", 0.0))),
                peak_id=idx,
            )
            for idx, peak in enumerate(peak_values)
        ]
        names = compound.get("names", [])
        return cls(
            precursor_mz=precursor_mz,
            precursor_formula=compound.get("formula"),
            precursor_ion=precursor_ion,
            peaks=peaks,
            name=names[0] if names else record.get("title"),
            accession=record.get("accession"),
            metadata={
                "title": record.get("title"),
                "splash": record.get("peak", {}).get("splash"),
            },
        )

