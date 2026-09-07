#!/usr/bin/env python3
"""Train the ordinary Mixed-AT baseline on all three peer attack families."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.aes_rhi_training_launcher import main

if __name__ == "__main__":
    raise SystemExit(main("mixed_at_rhi"))
