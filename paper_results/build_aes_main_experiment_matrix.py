#!/usr/bin/env python3
"""Collect completed AES attack/defense JSON files into paper-ready tables.

Rudimentary, HotFlip, and Injection are the three peer attack--defense
families. MLM-guided is reported separately as attack-only transfer. Missing
experiments remain null/blank and are never silently converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model": "B0",
        "training_exposure": "none",
        "clean": "eval_b0_seed42",
        "rudimentary": "eval_rudimentary_b0_seed42",
        "hotflip": "eval_b0_seed42",
        "injection": "eval_injection_b0_seed42",
        "mlm_guided": "eval_mlm_b0_seed42",
    },
    {
        "model": "C0",
        "training_exposure": "clean continuation",
        "clean": "eval_c0_seed42",
        "rudimentary": "eval_rudimentary_c0_seed42",
        "hotflip": "eval_c0_seed42",
        "injection": "eval_injection_c0_seed42",
        "mlm_guided": "eval_mlm_c0_seed42",
    },
    {
        "model": "D-HotFlip",
        "training_exposure": "hotflip",
        "clean": "eval_hotflip_defense_selected_seed42",
        "rudimentary": "eval_rudimentary_hotflip_defense_seed42",
        "hotflip": "eval_hotflip_defense_selected_seed42",
        "injection": "eval_injection_hotflip_defense_seed42",
        "mlm_guided": "eval_mlm_hotflip_defense_seed42",
    },
    {
        "model": "D-Rudimentary-v2",
        "training_exposure": "rudimentary",
        "clean": "eval_rudimentary_defense_v2_seed42",
        "rudimentary": "eval_rudimentary_defense_v2_seed42",
        "hotflip": "eval_hotflip_rudimentary_defense_v2_seed42",
        "injection": "eval_injection_rudimentary_defense_v2_seed42",
        "mlm_guided": "eval_mlm_rudimentary_defense_v2_seed42",
    },
    {
        "model": "D-Injection",
        "training_exposure": "injection",
        "clean": "eval_injection_defense_seed42",
        "rudimentary": "eval_rudimentary_injection_defense_seed42",
        "hotflip": "eval_hotflip_injection_defense_seed42",
        "injection": "eval_injection_defense_seed42",
        "mlm_guided": "eval_mlm_injection_defense_seed42",
    },
    {
        "model": "Mixed-AT-RH",
        "training_exposure": "rudimentary+hotflip",
        "clean": "eval_rudimentary_mixed_at_rh_seed42",
        "rudimentary": "eval_rudimentary_mixed_at_rh_seed42",
        "hotflip": "eval_hotflip_mixed_at_rh_seed42",
        "injection": "eval_injection_mixed_at_rh_seed42",
        "mlm_guided": "eval_mlm_mixed_at_rh_seed42",
    },
    {
        "model": "PAER-RH-v1",
        "training_exposure": "rudimentary+hotflip",
        "clean": "eval_rudimentary_paer_rh_seed42",
        "rudimentary": "eval_rudimentary_paer_rh_seed42",
        "hotflip": "eval_hotflip_paer_rh_seed42",
        "injection": "eval_injection_paer_rh_seed42",
        "mlm_guided": "eval_mlm_paer_rh_seed42",
    },
    {
        "model": "PAER-RH-v2",
        "training_exposure": "rudimentary+hotflip",
        "clean": "eval_rudimentary_paer_rh_v2_seed42",
        "rudimentary": "eval_rudimentary_paer_rh_v2_seed42",
        "hotflip": "eval_hotflip_paer_rh_v2_seed42",
        "injection": "eval_injection_paer_rh_v2_seed42",
        "mlm_guided": "eval_mlm_paer_rh_v2_seed42",
    },
    {
        "model": "PAER-RH-v3",
        "training_exposure": "rudimentary+hotflip",
        "clean": "eval_rudimentary_paer_rh_v3_seed42",
        "rudimentary": "eval_rudimentary_paer_rh_v3_seed42",
        "hotflip": "eval_hotflip_paer_rh_v3_seed42",
        "injection": "eval_injection_paer_rh_v3_seed42",
        "mlm_guided": "eval_mlm_paer_rh_v3_seed42",
    },
)


FIELDS = (
    "model",
    "training_exposure",
    "clean_qwk",
    "clean_mae",
    "rudimentary_asr",
    "rudimentary_avg_delta",
    "hotflip_asr",
    "hotflip_avg_delta",
    "injection_asr",
    "injection_avg_delta",
    "rhi_macro_asr",
    "mlm_guided_asr",
    "mlm_guided_avg_delta",
)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_metrics(directory: Path) -> dict[str, float | None]:
    payload = _read_json(directory / "clean_qwk.json")
    if not isinstance(payload, dict):
        return {"qwk": None, "mae": None}
    return {
        "qwk": _optional_float(payload.get("qwk")),
        "mae": _optional_float(payload.get("mae")),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _attack_metrics(directory: Path, attack: str) -> dict[str, float | None]:
    payload = _read_json(directory / "asr_summary.json")
    if not isinstance(payload, list):
        return {"asr": None, "avg_delta": None}
    row = next(
        (
            item
            for item in payload
            if isinstance(item, dict) and item.get("attack") == attack
        ),
        None,
    )
    if row is None:
        return {"asr": None, "avg_delta": None}
    return {
        "asr": _optional_float(row.get("asr")),
        "avg_delta": _optional_float(row.get("avg_delta")),
    }


def _injection_metrics(directory: Path) -> dict[str, float | None]:
    payload = _read_json(directory / "injection_family_summary.json")
    if not isinstance(payload, dict):
        return {"asr": None, "avg_delta": None}
    return {
        "asr": _optional_float(payload.get("asr")),
        "avg_delta": _optional_float(payload.get("avg_delta")),
    }


def collect_rows(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        clean = _clean_metrics(outputs_dir / spec["clean"])
        rudimentary = _attack_metrics(
            outputs_dir / spec["rudimentary"], "rudimentary"
        )
        hotflip = _attack_metrics(outputs_dir / spec["hotflip"], "hotflip")
        injection = _injection_metrics(outputs_dir / spec["injection"])
        mlm = _attack_metrics(
            outputs_dir / spec["mlm_guided"], "mlm_guided"
        )
        primary_asrs = (
            rudimentary["asr"],
            hotflip["asr"],
            injection["asr"],
        )
        rhi_macro = (
            sum(float(value) for value in primary_asrs) / 3.0
            if all(value is not None for value in primary_asrs)
            else None
        )
        rows.append(
            {
                "model": spec["model"],
                "training_exposure": spec["training_exposure"],
                "clean_qwk": clean["qwk"],
                "clean_mae": clean["mae"],
                "rudimentary_asr": rudimentary["asr"],
                "rudimentary_avg_delta": rudimentary["avg_delta"],
                "hotflip_asr": hotflip["asr"],
                "hotflip_avg_delta": hotflip["avg_delta"],
                "injection_asr": injection["asr"],
                "injection_avg_delta": injection["avg_delta"],
                "rhi_macro_asr": rhi_macro,
                "mlm_guided_asr": mlm["asr"],
                "mlm_guided_avg_delta": mlm["avg_delta"],
            }
        )
    return rows


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        ("Model", "model"),
        ("Training exposure", "training_exposure"),
        ("Clean QWK", "clean_qwk"),
        ("Rudi ASR", "rudimentary_asr"),
        ("HotFlip ASR", "hotflip_asr"),
        ("Injection ASR", "injection_asr"),
        ("RHI Macro ASR", "rhi_macro_asr"),
        ("MLM ASR (attack-only)", "mlm_guided_asr"),
    )
    lines = [
        "# AES main experiment matrix (seed 42)",
        "",
        (
            "Rudimentary, HotFlip, and Injection are peer attack--defense "
            "families. MLM-guided is attack-only transfer evaluation."
        ),
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format(row[key]) for _, key in columns)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build JSON/CSV/Markdown tables from completed AES outputs."
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=REPO_ROOT / "outputs"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "outputs" / "paper_results"
    )
    args = parser.parse_args()
    rows = collect_rows(args.outputs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "aes_main_experiment_matrix_seed42.json"
    csv_path = args.out_dir / "aes_main_experiment_matrix_seed42.csv"
    markdown_path = args.out_dir / "aes_main_experiment_matrix_seed42.md"
    json_path.write_text(
        json.dumps(
            {
                "taxonomy": {
                    "peer_attack_defense_families": [
                        "rudimentary",
                        "hotflip",
                        "injection",
                    ],
                    "attack_only_transfer": "mlm_guided",
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(markdown_path, rows)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
