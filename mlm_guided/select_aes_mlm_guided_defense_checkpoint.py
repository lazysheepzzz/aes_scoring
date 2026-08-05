#!/usr/bin/env python3
"""Select D_MLM using the clean-QWK gate and matching subset ASR."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox.select_aes_hotflip_defense_checkpoint import main


if __name__ == "__main__":
    raise SystemExit(main("mlm_guided"))
