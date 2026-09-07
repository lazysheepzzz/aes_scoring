#!/usr/bin/env python3
"""Fill the MLM transfer column for D-HotFlip, D-Rudimentary-v2 and D-Injection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.evaluate_aes_rhi_experiments import main

if __name__ == "__main__":
    raise SystemExit(main("specialists-mlm"))
