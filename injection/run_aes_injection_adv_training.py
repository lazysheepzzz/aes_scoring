#!/usr/bin/env python3
"""Train the original-paper-style D_INJECTION AES defense baseline."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox.aes_stage2_training_launcher import INJECTION_DEFENSE, main


if __name__ == "__main__":
    raise SystemExit(main(INJECTION_DEFENSE))
