#!/usr/bin/env python3
"""Launch AES attack evaluation without relying on a shell script.

The positional interface remains compatible with the former launcher:

    python whitebox/run_aes_attacks.py [attack_name] [n_essays]

Paths and runtime settings can also be overridden with named arguments.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SERVER_ROOT = Path("/root/autodl-tmp")
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_REPO_ROOT = SERVER_ROOT / "robust_text_scoring"
REPO_ROOT = (
    SERVER_REPO_ROOT
    if (SERVER_REPO_ROOT / "text_scoring_adv_training").is_dir()
    else SCRIPT_REPO_ROOT
)
ON_SERVER = SERVER_ROOT.exists()
ATTACKS = ("rudimentary", "injection", "hotflip", "mlm_guided", "all")


def _default_path(server_path: str, local_path: Path) -> Path:
    return Path(server_path) if ON_SERVER else local_path


def _python_prefix(conda_env: str | None, current_python: bool) -> list[str]:
    if current_python or not conda_env:
        return [sys.executable]
    if os.environ.get("CONDA_DEFAULT_ENV") == conda_env:
        return [sys.executable]

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    server_conda = Path("/root/miniconda3/bin/conda")
    if not conda and server_conda.is_file():
        conda = str(server_conda)
    if conda:
        return [conda, "run", "--no-capture-output", "-n", conda_env, "python"]

    print(
        f"[WARN] Conda environment {conda_env!r} was requested but conda was not found; "
        f"using {sys.executable}.",
        file=sys.stderr,
    )
    return [sys.executable]


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)


def _configure_environment(args: argparse.Namespace) -> None:
    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.hf_home)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AES adversarial attacks.")
    parser.add_argument("attack_name", nargs="?", choices=ATTACKS)
    parser.add_argument("essay_count", nargs="?", type=int)
    parser.add_argument("--attack", dest="attack_option", choices=ATTACKS)
    parser.add_argument("--n-essays", dest="n_essays_option", type=int)
    parser.add_argument(
        "--victim",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/victim/fold0_best",
            REPO_ROOT / "deberta_checkpoints" / "fold0_best",
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/data/valid_fold0.csv",
            REPO_ROOT / "data" / "valid_fold0.csv",
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/aes_results",
            REPO_ROOT / "outputs" / "aes_results",
        ),
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=_default_path(
            "/root/autodl-tmp/hf_cache",
            REPO_ROOT / ".cache" / "huggingface",
        ),
    )
    parser.add_argument(
        "--conda-env",
        default="aes" if ON_SERVER else None,
        help="Conda environment used to run the evaluator. Server default: aes.",
    )
    parser.add_argument(
        "--current-python",
        action="store_true",
        help="Use the current Python interpreter instead of conda run.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face network access. Offline mode is the default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running an attack.",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    attack = args.attack_option or args.attack_name or "all"
    n_essays = (
        args.n_essays_option
        if args.n_essays_option is not None
        else args.essay_count
        if args.essay_count is not None
        else 200
    )
    if n_essays <= 0:
        raise ValueError("n_essays must be greater than zero")

    entrypoint = (
        REPO_ROOT
        / "text_scoring_adv_training"
        / "evaluation"
        / "aes"
        / "run_attacks.py"
    )
    command = [
        *_python_prefix(args.conda_env, args.current_python),
        str(entrypoint),
        "--victim",
        str(args.victim),
        "--data",
        str(args.data),
        "--attack",
        attack,
        "--n-essays",
        str(n_essays),
        "--out",
        str(args.out),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.thresholds is not None:
        command.extend(["--thresholds", str(args.thresholds)])
    return command


def main() -> int:
    args = build_parser().parse_args()
    _configure_environment(args)
    command = build_command(args)

    print(f"[ENV] python launcher: {_format_command(command[:1])}")
    print(f"[ENV] HF_HOME: {os.environ['HF_HOME']}")
    print(f"[RUN] {_format_command(command)}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, env=os.environ.copy(), check=False)
    if completed.returncode == 0:
        print(f"[DONE] Results in {args.out}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
