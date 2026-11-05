"""Matched-count, fixed per-state negative control for v3 localization."""
from __future__ import annotations

import hashlib
import random


def shuffled_targets(targets: list[float], visible: list[bool], *, seed: int,
                     row_index: int, adversarial_text: str) -> list[float]:
    """Uniformly relocate positive values among eligible tokens, without replacement.

    Chance overlap with true positions is allowed. State-keyed randomness leaves
    dataloader/model RNG untouched and does not vary on subsequent exposures.
    """
    if len(targets) != len(visible) or any(t != 0 and not v for t, v in zip(targets, visible)):
        raise ValueError("Targets and visible-token mask do not agree")
    positive = [value for value in targets if value > 0]
    if not positive:
        return targets.copy()
    eligible = [i for i, value in enumerate(visible) if value]
    state = hashlib.sha256(adversarial_text.encode("utf-8")).hexdigest()
    rng = random.Random(f"paer-shuffled-localization-v1:{seed}:{row_index}:{state}")
    result = [0.0] * len(targets)
    for index, value in zip(rng.sample(eligible, len(positive)), positive):
        result[index] = value
    return result
