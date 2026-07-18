"""
InjectionAttack: sentence injection and sentence duplication attacks.
v1: mode="injection", position="random", n_variants=1.
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
