"""Train the expert-only RHI control in a NEW directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer.expert_routing.launch import main, EXPERT_ONLY

if __name__ == '__main__':
    raise SystemExit(main(EXPERT_ONLY))
