#!/usr/bin/env python3
"""Train the same-data Mixed-AT-RH structural control baseline."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paer.aes_rh_training_launcher import MIXED_AT_RH, main


if __name__ == "__main__":
    raise SystemExit(main(MIXED_AT_RH))
