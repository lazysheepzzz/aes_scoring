"""
RudimentaryAttack: character-level and word-level random edits.

``RudimentaryAttack`` preserves the paper's simple variant generator.
``IterativeRudimentaryAttack`` is the AES evaluation wrapper that performs
budgeted greedy search and true scorer validation.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

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

__all__ = ["RudimentaryAttack", "IterativeRudimentaryAttack"]


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


class IterativeRudimentaryAttack:
    """Greedy AES Rudimentary search under a fixed edit-operation budget.

    The underlying character/word edits remain the original paper operations.
    At every search step this wrapper generates multiple variants, scores the
    actual candidate texts with the AES scorer, and accepts only the
    highest-scoring improvement.
    """

    def __init__(
        self,
        scorer,
        *,
        n_steps: int = 30,
        beam_size: int = 1,
        candidates_per_step: int = 16,
        batch_size: int = 32,
        threshold: float = 0.1,
        max_token_edit_rate: float | None = 0.1,
        improvement_tolerance: float = 1e-6,
        record_intermediate_texts: bool = False,
    ):
        if n_steps <= 0:
            raise ValueError("n_steps must be greater than zero")
        if beam_size != 1:
            raise ValueError(
                "IterativeRudimentaryAttack currently supports beam_size=1"
            )
        if candidates_per_step <= 0:
            raise ValueError("candidates_per_step must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if (
            max_token_edit_rate is not None
            and not 0 < max_token_edit_rate <= 1
        ):
            raise ValueError("max_token_edit_rate must be in (0, 1]")
        if improvement_tolerance < 0:
            raise ValueError("improvement_tolerance must be non-negative")

        self.scorer = scorer
        self.n_steps = n_steps
        self.beam_size = beam_size
        self.candidates_per_step = candidates_per_step
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_token_edit_rate = max_token_edit_rate
        self.improvement_tolerance = improvement_tolerance
        self.record_intermediate_texts = record_intermediate_texts

    def _token_ids(self, text: str) -> Tuple[int, ...]:
        encoded = self.scorer.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return tuple(int(token_id) for token_id in input_ids)

    @staticmethod
    def _describe_change(before: str, after: str) -> List[Dict[str, Any]]:
        """Store compact character spans instead of repeating full essays."""
        changes: List[Dict[str, Any]] = []
        matcher = SequenceMatcher(a=before, b=after, autojunk=False)
        for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            changes.append(
                {
                    "type": tag,
                    "before_start": before_start,
                    "before_end": before_end,
                    "after_start": after_start,
                    "after_end": after_end,
                    "deleted": before[before_start:before_end],
                    "inserted": after[after_start:after_end],
                }
            )
        return changes

    def _edit_budget(self, text: str) -> int:
        if self.max_token_edit_rate is None:
            return self.n_steps
        return min(
            self.n_steps,
            int(len(self._token_ids(text)) * self.max_token_edit_rate),
        )

    def attack(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        original_score = float(self.scorer.score_single(text))
        best_text = text
        best_score = original_score
        history: List[Dict[str, Any]] = []
        visited = {text}
        best_token_ids = self._token_ids(text)
        edit_budget = self._edit_budget(text)
        if edit_budget <= 0:
            return text, history

        accepted_edits = 0
        for step in range(self.n_steps):
            if accepted_edits >= edit_budget:
                break

            generated_candidates = sample_variants(
                best_text,
                self.candidates_per_step,
            )
            candidates: List[str] = []
            candidate_token_ids: List[Tuple[int, ...]] = []
            for candidate in generated_candidates:
                if (
                    not candidate.strip()
                    or candidate in visited
                ):
                    continue
                visited.add(candidate)
                token_ids = self._token_ids(candidate)
                if token_ids == best_token_ids:
                    continue
                candidates.append(candidate)
                candidate_token_ids.append(token_ids)
            if not candidates:
                continue

            scores = self.scorer.score_batch(
                candidates,
                batch_size=self.batch_size,
            )
            best_index = max(
                range(len(candidates)),
                key=lambda index: float(scores[index]),
            )
            candidate_text = candidates[best_index]
            candidate_score = float(scores[best_index])

            if candidate_score > best_score + self.improvement_tolerance:
                previous_text = best_text
                previous_score = best_score
                best_text = candidate_text
                best_token_ids = candidate_token_ids[best_index]
                best_score = candidate_score
                accepted_edits += 1
                history_entry = {
                        "step": step,
                        "score": best_score,
                        "step_gain": best_score - previous_score,
                        "delta": best_score - original_score,
                        "accepted_edit_count": accepted_edits,
                        "max_edits": edit_budget,
                        "changes": self._describe_change(
                            previous_text,
                            best_text,
                        ),
                    }
                if self.record_intermediate_texts:
                    history_entry["before_text"] = previous_text
                    history_entry["after_text"] = best_text
                history.append(history_entry)

            if best_score - original_score >= self.threshold:
                break

        return best_text, history

    def attack_batch(
        self,
        texts: List[str],
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [self.attack(text) for text in texts]
