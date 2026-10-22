"""Read-only request/budget preview; this entry point cannot upload essays."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.training_quality_pairs import preview_main

if __name__ == '__main__':
    preview_main()
