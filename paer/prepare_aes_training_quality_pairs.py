"""Export only existing training steps; this entry point cannot call an API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.training_quality_pairs import prepare_main

if __name__ == '__main__':
    prepare_main()
