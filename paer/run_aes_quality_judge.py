"""Run cached, budget-limited DeepSeek essay quality judgments, not training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.quality_relation_audit import judge_main

if __name__ == "__main__":
    judge_main()
