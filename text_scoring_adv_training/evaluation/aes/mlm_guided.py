"""Backward-compatible import for the formal AES MLM-guided attack."""

from text_scoring_adv_training.evaluation.aes.attacks.mlm_guided import (
    MLMGuidedAttack,
    MLMGuidedCandidateGenerator,
    SemanticSimilarityFilter,
)

__all__ = [
    "MLMGuidedAttack",
    "MLMGuidedCandidateGenerator",
    "SemanticSimilarityFilter",
]
