"""
InjectionAttack: sentence injection and sentence duplication attacks.
v1: mode="injection", position="random", n_variants=1.
"""
from __future__ import annotations

import random
import re
import sys
import os
from typing import Any, List

__all__ = ["InjectionAttack", "IterativeInjectionAttack"]


IRRELEVANT_SENTENCES = [
    "The city of Melbourne is known for its vibrant coffee culture.",
    "A group of scientists recently discovered a new species of deep-sea fish.",
    "Historical records indicate that the library of Alexandria was destroyed multiple times.",
    "The inventor of the printing press revolutionized the spread of information.",
    "Mountains form through tectonic plate collisions over millions of years.",
    "A well-balanced diet includes a variety of fruits and vegetables.",
    "The industrial revolution began in Britain during the eighteenth century.",
    "Astronomers use telescopes to observe distant galaxies and stars.",
    "Ancient Egyptians built the pyramids as tombs for their pharaohs.",
    "The human brain contains approximately eighty-six billion neurons.",
    "Water freezes at zero degrees Celsius at standard atmospheric pressure.",
    "Bees are essential pollinators for many crops and wild plants.",
    "The Amazon River is the largest river by discharge volume in the world.",
    "Classical music composers like Mozart wrote hundreds of symphonies.",
    "The human skeleton consists of 206 bones in an adult body.",
    "Photography emerged as an art form in the early nineteenth century.",
    "Carbon dioxide is a greenhouse gas that contributes to global warming.",
    "The telephone was invented by Alexander Graham Bell in 1876.",
    "Solar panels convert sunlight into electrical energy using photovoltaic cells.",
    "The immune system protects the body against infections and diseases.",
    "Shakespeare wrote tragedies, comedies, and histories during the Elizabethan era.",
    "Glaciers store about two-thirds of the world's fresh water.",
    "The internet consists of millions of interconnected computer networks.",
    "Honey bees can communicate the location of food sources through dance.",
    "The principles of supply and demand influence market prices.",
    "Penguins are flightless birds that primarily live in the Southern Hemisphere.",
    "The Magna Carta was signed in 1215 and limited the power of the king.",
    "Electric vehicles produce zero direct exhaust emissions during operation.",
    "The ozone layer shields the Earth from harmful ultraviolet radiation.",
    "DNA carries the genetic instructions for the development and functioning of living organisms.",
]


class InjectionAttack:
    """
    Sentence injection and duplication attack.
    """

    def __init__(
        self,
        scorer=None,
        sentence_bank: List[str] = None,
        *,
        mode: str = "injection",
        position: str = "random",
        n_variants: int = 1,
    ):
        self.scorer = scorer
        self.sentence_bank = sentence_bank or IRRELEVANT_SENTENCES
        self.mode = mode
        self.position = position
        self.n_variants = n_variants

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _inject(self, text: str) -> str:
        sentences = self._split_sentences(text)
        inject_sentence = random.choice(self.sentence_bank)

        pos = self.position
        if pos == "random":
            pos = random.choice(["start", "middle", "end"])

        if pos == "start":
            return f"{inject_sentence} {' '.join(sentences)}"
        elif pos == "end":
            return f"{' '.join(sentences)} {inject_sentence}"
        else:  # middle
            if len(sentences) <= 1:
                return f"{' '.join(sentences)} {inject_sentence}"
            idx = random.randint(1, len(sentences) - 1)
            return " ".join(sentences[:idx] + [inject_sentence] + sentences[idx:])

    def _duplicate(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text
        src = random.choice(sentences)
        pos = self.position
        if pos == "random":
            pos = random.choice(["start", "middle", "end"])
        if pos == "start":
            return f"{src} {' '.join(sentences)}"
        elif pos == "end":
            return f"{' '.join(sentences)} {src}"
        else:
            if len(sentences) <= 1:
                return f"{' '.join(sentences)} {src}"
            idx = random.randint(1, len(sentences) - 1)
            return " ".join(sentences[:idx] + [src] + sentences[idx:])

    def attack(self, text: str) -> List[str]:
        variants = []
        for _ in range(self.n_variants):
            if self.mode == "duplication":
                variants.append(self._duplicate(text))
            else:
                variants.append(self._inject(text))
        return variants

    def attack_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.attack(t) for t in texts]


class IterativeInjectionAttack:
    """Scorer-guided AES sentence injection used by the formal protocol.

    This AES adapter follows the existing experiment scripts without changing
    the original repository's shared injection utilities.  External sentence
    injection and self-duplication are deliberately separate attack modes.
    """

    MODES = ("external", "self_duplication")

    def __init__(
        self,
        scorer,
        *,
        mode: str,
        sentence_bank: List[str] | None = None,
        n_steps: int = 30,
        candidates_per_step: int = 16,
        batch_size: int = 4,
        threshold: float = 0.1,
        improvement_tolerance: float = 1e-6,
        record_intermediate_texts: bool = True,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown injection mode: {mode}")
        if n_steps <= 0 or candidates_per_step <= 0 or batch_size <= 0:
            raise ValueError(
                "n_steps, candidates_per_step, and batch_size must be positive"
            )
        if threshold < 0 or improvement_tolerance < 0:
            raise ValueError("threshold and improvement_tolerance must be non-negative")
        if mode == "external" and not sentence_bank:
            raise ValueError("external injection requires a non-empty sentence bank")
        self.scorer = scorer
        self.mode = mode
        self.sentence_bank = list(sentence_bank or [])
        self.n_steps = n_steps
        self.candidates_per_step = candidates_per_step
        self.batch_size = batch_size
        self.threshold = threshold
        self.improvement_tolerance = improvement_tolerance
        self.record_intermediate_texts = record_intermediate_texts

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence for sentence in sentences if sentence.strip()]

    @staticmethod
    def _grid_dimensions(
        n_sources: int,
        n_destinations: int,
        budget: int,
    ) -> tuple[int, int]:
        """Choose a near-square source/destination grid within the budget."""
        source_count = min(n_sources, max(1, int(budget**0.5)))
        destination_count = min(
            n_destinations,
            max(1, budget // source_count),
        )
        return source_count, destination_count

    def _external_candidates(self, text: str) -> list[dict[str, Any]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        destinations = list(range(len(sentences) + 1))
        n_sources, n_destinations = self._grid_dimensions(
            len(self.sentence_bank),
            len(destinations),
            self.candidates_per_step,
        )
        sampled_sentences = random.sample(self.sentence_bank, n_sources)
        sampled_destinations = random.sample(destinations, n_destinations)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for injected_sentence in sampled_sentences:
            for destination in sampled_destinations:
                new_sentences = list(sentences)
                new_sentences.insert(destination, injected_sentence)
                candidate_text = " ".join(new_sentences)
                if candidate_text == text or candidate_text in seen:
                    continue
                seen.add(candidate_text)
                candidates.append(
                    {
                        "text": candidate_text,
                        "injected_sentence": injected_sentence,
                        "source_index": None,
                        "destination_index": destination,
                    }
                )
                if len(candidates) >= self.candidates_per_step:
                    return candidates
        return candidates

    def _self_duplication_candidates(self, text: str) -> list[dict[str, Any]]:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return []
        source_indices = list(range(len(sentences)))
        destinations = list(range(len(sentences) + 1))
        n_sources, n_destinations = self._grid_dimensions(
            len(source_indices),
            len(destinations),
            self.candidates_per_step,
        )
        sampled_sources = random.sample(source_indices, n_sources)
        sampled_destinations = random.sample(destinations, n_destinations)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_index in sampled_sources:
            source_sentence = sentences[source_index]
            for destination in sampled_destinations:
                if destination in {source_index, source_index + 1}:
                    continue
                new_sentences = list(sentences)
                new_sentences.insert(destination, source_sentence)
                candidate_text = " ".join(new_sentences)
                if candidate_text == text or candidate_text in seen:
                    continue
                seen.add(candidate_text)
                candidates.append(
                    {
                        "text": candidate_text,
                        "injected_sentence": source_sentence,
                        "source_index": source_index,
                        "destination_index": destination,
                    }
                )
                if len(candidates) >= self.candidates_per_step:
                    return candidates
        return candidates

    def build_candidates(self, text: str) -> list[dict[str, Any]]:
        if self.mode == "external":
            return self._external_candidates(text)
        return self._self_duplication_candidates(text)

    def attack(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        original_score = float(self.scorer.score_single(text))
        best_text = text
        best_score = original_score
        history: list[dict[str, Any]] = []

        for step in range(self.n_steps):
            candidates = self.build_candidates(best_text)
            if not candidates:
                break
            scores = self.scorer.score_batch(
                [candidate["text"] for candidate in candidates],
                batch_size=self.batch_size,
            )
            best_index = max(range(len(scores)), key=lambda index: scores[index])
            candidate = candidates[best_index]
            candidate_score = float(scores[best_index])
            if candidate_score > best_score + self.improvement_tolerance:
                previous_text = best_text
                step_gain = candidate_score - best_score
                best_text = str(candidate["text"])
                best_score = candidate_score
                entry = {
                    "step": step,
                    "score": best_score,
                    "step_gain": step_gain,
                    "delta": best_score - original_score,
                    "injection_mode": self.mode,
                    "injected_sentence": candidate["injected_sentence"],
                    "source_index": candidate["source_index"],
                    "destination_index": candidate["destination_index"],
                    "accepted_injection_count": len(history) + 1,
                }
                if self.record_intermediate_texts:
                    entry["before_text"] = previous_text
                    entry["after_text"] = best_text
                history.append(entry)
            if best_score - original_score >= self.threshold:
                break
        return best_text, history

    def attack_batch(self, texts: List[str]):
        return [self.attack(text) for text in texts]
