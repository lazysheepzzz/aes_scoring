"""
RudimentaryAttack: character-level and word-level random edits.
Tests resilience to typographic noise and light paraphrasing.

Strategy: generate MANY random variants and return ALL of them.
ASR = fraction of essays where ANY variant scores higher than original.
More variants = higher ASR (more attack surface).
"""
from __future__ import annotations

from typing import List

import sys as _sys
import os as _os

_project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import (
    sample_variants,
    sample_variants_with_frozen,
)

__all__ = ["RudimentaryAttack"]


class RudimentaryAttack:
    """
    Rudimentary character/word-level edit attack.

    Generates MANY perturbed variants per essay (20) and returns ALL of them.
    ASR = fraction of essays where any variant scores higher than original.
    """

    def __init__(
        self,
        scorer: "AESScorer" = None,    # noqa: F821
        *,
        n_variants: int = 20,
        frozen_prefix: str = "",
    ):
        """
        Args:
            scorer: AESScorer instance (not required, kept for compat).
            n_variants: number of perturbed variants to generate per essay.
            frozen_prefix: if non-empty, only edit text after this prefix.
        """
        self.scorer = scorer
        self.n_variants = n_variants
        self.frozen_prefix = frozen_prefix

    def attack(self, text: str) -> List[str]:
        """Generate n_variants perturbed variants (all of them)."""
        if self.frozen_prefix:
            return sample_variants_with_frozen(text, self.n_variants, self.frozen_prefix)
        return sample_variants(text, self.n_variants)

    def attack_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.attack(t) for t in texts]
