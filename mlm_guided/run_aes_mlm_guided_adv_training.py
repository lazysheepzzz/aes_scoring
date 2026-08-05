#!/usr/bin/env python3
"""Train the D_MLM quality-preserving adversarial defense."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox.aes_stage2_training_launcher import MLM_GUIDED_DEFENSE, main


if __name__ == "__main__":
    raise SystemExit(main(MLM_GUIDED_DEFENSE))
