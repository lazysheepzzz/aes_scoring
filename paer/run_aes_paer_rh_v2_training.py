#!/usr/bin/env python3
"""Train PAER-v2 with calibrated sparse directional evidence routing."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paer.aes_rh_training_launcher import PAER_RH_V2, main


if __name__ == "__main__":
    raise SystemExit(main(PAER_RH_V2))
