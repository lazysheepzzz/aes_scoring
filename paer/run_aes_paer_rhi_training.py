#!/usr/bin/env python3
"""Train the existing PAER-v3 architecture on shared R/H/I traces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.aes_rhi_training_launcher import main

if __name__ == "__main__":
    raise SystemExit(main("paer_rhi_v3"))
