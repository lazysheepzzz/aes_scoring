"""Export frozen AES attack pairs for blinded LLM quality assessment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.quality_relation_audit import prepare_main

if __name__ == "__main__":
    prepare_main()
