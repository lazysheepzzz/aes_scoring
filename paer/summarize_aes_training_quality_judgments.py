"""Offline summary of the NEW training-edit audit; never uploads essays."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.training_quality_judge import main

if __name__ == '__main__':
    raise SystemExit(main(summarize_only=True))
