"""
MLMGuidedAttack: WordNet synonym replacement + scorer validation.

Strategy: generate MANY synonym replacements and return ALL variants.
ASR = fraction of essays where ANY variant scores higher than original.
More candidates (lower sim_threshold) + more variants = higher ASR.
"""
from __future__ import annotations

import random
import re
import sys
import os
from typing import Dict, List, Optional, Set, Tuple

__all__ = ["MLMGuidedAttack"]


try:
    import nltk
    from nltk.corpus import wordnet as wn
    _HAS_WORDNET = True
    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
except Exception:
    _HAS_WORDNET = False


_POS_MAP = {
    "NN": wn.NOUN, "NNS": wn.NOUN, "NNP": wn.NOUN, "NNPS": wn.NOUN,
    "VB": wn.VERB, "VBD": wn.VERB, "VBG": wn.VERB, "VBN": wn.VERB,
    "VBP": wn.VERB, "VBZ": wn.VERB,
    "JJ": wn.ADJ, "JJR": wn.ADJ, "JJS": wn.ADJ,
    "RB": wn.ADV, "RBR": wn.ADV, "RBS": wn.ADV,
}


class MLMGuidedAttack:
    """
    WordNet-based synonym replacement attack.
    Returns ALL generated variants (not just the best) to maximize ASR.
    """

    def __init__(
        self,
        scorer: "AESScorer" = None,   # noqa: F821
        *,
        sim_threshold: float = 0.5,
        n_synonyms: int = 10,
        n_variants: int = 30,
        pos_tagger=None,
    ):
        """
        Args:
            scorer: AESScorer instance for scoring variants.
            sim_threshold: minimum WordNet similarity (0.5; lower = more candidates).
            n_synonyms: max synonyms to try per word (10).
            n_variants: max number of variants to return per essay (30).
            pos_tagger: optional pre-initialized NLTK pos tagger.
        """
        if not _HAS_WORDNET:
            raise ImportError("NLTK WordNet is not available.")
        self.scorer = scorer
        self.sim_threshold = sim_threshold
        self.n_synonyms = n_synonyms
        self.n_variants = n_variants
        self._pos_tagger = pos_tagger

    def _pos_to_wn(self, pos: str) -> Optional[object]:
        return _POS_MAP.get(pos)

    def _get_synonyms(self, word: str, pos: str) -> List[Tuple[str, float]]:
        """Return list of (synonym_word, similarity_score)."""
        wn_pos = self._pos_to_wn(pos)
        if wn_pos is None:
            return []

        synsets = wn.synsets(word, pos=wn_pos)
        if not synsets:
            return []

        anchor = synsets[0]
        candidates: List[Tuple[str, float]] = []

        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() == word.lower().replace("_", " "):
                    continue
                try:
                    sim = synset.path_similarity(anchor)
                    if sim is not None and sim >= self.sim_threshold:
                        candidates.append((lemma.name().replace("_", " "), sim))
                except Exception:
                    continue

        seen: Set[str] = set()
        unique: List[Tuple[str, float]] = []
        for cand, sim in sorted(candidates, key=lambda x: -x[1]):
            if cand.lower() not in seen:
                seen.add(cand.lower())
                unique.append((cand, sim))
                if len(unique) >= self.n_synonyms:
                    break

        return unique

    @staticmethod
    def _tokenize_words(text: str) -> List[Tuple[str, str]]:
        """Tokenize with NLTK POS tagging."""
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger", quiet=True)
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            return tagged
        except Exception:
            words = re.findall(r"[A-Za-z]+", text)
            return [(w, "NN") for w in words]

    def _build_replacement_candidates(self, text: str) -> List[Tuple[int, str, str]]:
        """Returns list of (word_index, synonym, original_word)."""
        tagged = self._tokenize_words(text)
        words = text.split()
        candidates: List[Tuple[int, str, str]] = []

        for idx, (word, pos) in enumerate(tagged):
            if idx >= len(words):
                break
            if pos in ("DT", "IN", "TO", "CC", "CD", "EX", "MD", "PDT", "POS", "PRP", "PRP$", "WDT", "WP", "WP$"):
                continue
            syns = self._get_synonyms(word, pos)
            for synonym, sim in syns:
                candidates.append((idx, synonym, words[idx]))

        return candidates

    def attack(self, text: str) -> List[str]:
        """
        Try all synonym replacements and return ALL variants (up to n_variants).
        Returns variants sorted by score descending.
        """
        if self.scorer is None:
            return []

        words = text.split()
        candidates = self._build_replacement_candidates(text)
        if not candidates:
            return []

        # Build all single-word replacement variants
        all_variants: List[Tuple[float, str]] = []
        seen: Set[str] = set()

        for idx, synonym, orig_word in candidates:
            new_words = words.copy()
            new_words[idx] = synonym
            new_text = " ".join(new_words)
            if new_text in seen:
                continue
            seen.add(new_text)
            try:
                score = self.scorer.score_single(new_text)
                all_variants.append((score, new_text))
            except Exception:
                continue

        if not all_variants:
            return []

        # Sort by score descending and return top n_variants
        all_variants.sort(key=lambda x: -x[0])
        return [v[1] for v in all_variants[: self.n_variants]]

    def attack_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.attack(t) for t in texts]
