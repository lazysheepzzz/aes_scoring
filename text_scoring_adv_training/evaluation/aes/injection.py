"""
InjectionAttack: sentence injection and sentence duplication attacks.
Tests whether lengthening / padding an essay with content can manipulate the AES score.

Strategy: generate MANY variants (different sentences + positions) and return ALL.
ASR = fraction of essays where ANY variant scores higher than original.
"""
from __future__ import annotations

import random
import re
import sys
import os
from typing import List

__all__ = ["InjectionAttack"]


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
    "The stock market fluctuates based on investor sentiment and economic data.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "The human digestive system breaks down food into absorbable nutrients.",
    "Ocean currents play a critical role in regulating global climate patterns.",
]


class InjectionAttack:
    """
    Sentence injection and duplication attack.
    Returns ALL generated variants (injection + duplication) for maximum ASR.
    """

    def __init__(
        self,
        scorer: "AESScorer" = None,   # noqa: F821
        sentence_bank: List[str] | None = None,
        *,
        n_variants: int = 20,
    ):
        """
        Args:
            scorer: AESScorer instance (not required, kept for compat).
            sentence_bank: list of irrelevant sentences.
            n_variants: total number of variants to generate per essay (20).
        """
        self.scorer = scorer
        self.sentence_bank = sentence_bank or IRRELEVANT_SENTENCES
        self.n_variants = n_variants

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _inject(self, text: str, sentence: str, pos: str) -> str:
        sentences = self._split_sentences(text)
        if pos == "start":
            return f"{sentence} {' '.join(sentences)}"
        elif pos == "end":
            return f"{' '.join(sentences)} {sentence}"
        else:
            if len(sentences) <= 1:
                return f"{' '.join(sentences)} {sentence}"
            idx = random.randint(1, len(sentences) - 1)
            return " ".join(sentences[:idx] + [sentence] + sentences[idx:])

    def _duplicate(self, text: str, src_idx: int, pos: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences or src_idx >= len(sentences):
            return text
        src = sentences[src_idx]
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
        """
        Generate n_variants different injection/duplication variants.
        Returns ALL of them (not just the best) to maximize ASR.
        """
        sentences = self._split_sentences(text)
        positions = ["start", "middle", "end"]
        variants = []
        used = set()

        half = self.n_variants // 2

        # Injection variants
        for _ in range(half):
            sentence = random.choice(self.sentence_bank)
            pos = random.choice(positions)
            v = self._inject(text, sentence, pos)
            if v not in used:
                used.add(v)
                variants.append(v)

        # Duplication variants
        if sentences:
            for _ in range(self.n_variants - half):
                src_idx = random.randint(0, len(sentences) - 1)
                pos = random.choice(positions)
                v = self._duplicate(text, src_idx, pos)
                if v not in used:
                    used.add(v)
                    variants.append(v)

        return variants

    def attack_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.attack(t) for t in texts]
