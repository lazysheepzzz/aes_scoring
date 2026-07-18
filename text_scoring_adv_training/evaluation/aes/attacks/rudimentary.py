"""
RudimentaryAttack: character-level and word-level random edits.
v1 parameters: n_variants=1 (simple random edit).
"""
from __future__ import annotations

from typing import List

import sys as _sys
import os as _os

_d = _os.path.dirname
_repo_root = _d(_d(_d(_d(_d(_os.path.abspath(__file__))))))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import (
    sample_variants,
    sample_variants_with_frozen,
)

__all__ = ["RudimentaryAttack"]


class RudimentaryAttack:
    """
    Rudimentary character/word-level edit attack.
    Generates n_variants random perturbations and returns all of them.
    """

    def __init__(
        self,
        scorer=None,
        *,
        n_variants: int = 1,
        frozen_prefix: str = "",
    ):
        self.scorer = scorer
        self.n_variants = n_variants
        self.frozen_prefix = frozen_prefix

    def attack(self, text: str) -> List[str]:
        if self.frozen_prefix:
            return sample_variants_with_frozen(text, self.n_variants, self.frozen_prefix)
        return sample_variants(text, self.n_variants)

    def attack_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.attack(t) for t in texts]
