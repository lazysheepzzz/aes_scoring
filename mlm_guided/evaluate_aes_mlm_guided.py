#!/usr/bin/env python3
"""Evaluate any AES checkpoint on clean metrics and formal MLM-guided ASR."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox.eval_hotflip_defended import main


if __name__ == "__main__":
    raise SystemExit(main("mlm_guided"))
