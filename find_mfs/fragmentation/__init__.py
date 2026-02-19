"""
Fragmentation tree construction via Maximum Colorful Subtree ILP.

Implements a SIRIUS-like approach for building fragmentation trees from
MS/MS spectra, with isotope-envelope-aware scoring.
"""
from .tree import (
    FragmentationTreeGenerator,
    FragmentationTree,
    FragmentNode,
    FragmentEdge,
    Peak,
    IsotopeEnvelope,
    detect_isotope_envelopes,
)

__all__ = [
    "FragmentationTreeGenerator",
    "FragmentationTree",
    "FragmentNode",
    "FragmentEdge",
    "Peak",
    "IsotopeEnvelope",
    "detect_isotope_envelopes",
]
