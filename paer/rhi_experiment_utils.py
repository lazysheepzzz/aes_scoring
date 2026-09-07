"""CPU-only provenance and output guards shared by new RHI entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RHI_ATTACKS = ("rudimentary", "hotflip", "injection_external", "injection_self_dup")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    path = Path(path)
    files = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin"))
                   | set(path.glob("paer_*.pt")) | set(path.glob("*.json")))
    if not any(p.suffix == ".safetensors" or p.name.startswith("pytorch_model") for p in files):
        raise FileNotFoundError(f"Model weights not found: {path}")
    return {p.name: sha256(p) for p in files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    """Atomic metadata update within an already bound new experiment directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bind_directory(path, protocol):
    """Resume only an identical experiment; never adopt an existing foreign output."""
    path = Path(path)
    manifest = path / "rhi_run_binding.json"
    if manifest.exists():
        if read_json(manifest) != protocol:
            raise ValueError(f"Protocol/input changed. Choose a NEW output directory: {path}")
    else:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to use nonempty unbound output: {path}")
        save_json(manifest, protocol)


def rhi_macro(rudimentary, hotflip, external, self_dup):
    return (rudimentary + hotflip + (external + self_dup) / 2.0) / 3.0


def validate_trace_groups(records, clean_rows):
    """One attack per essay makes rotating traces exactly family-balanced per epoch."""
    import math
    grouped = {}
    ids = set()
    for record in records:
        attack = record["attack"]
        if attack not in RHI_ATTACKS:
            raise ValueError(f"Forbidden training attack: {attack}")
        row_index = int(record["row_index"])
        if row_index < 0 or row_index >= len(clean_rows):
            raise ValueError(f"Trace row outside training CSV: {row_index}")
        row = clean_rows[row_index]
        if record["original_text"] != row["text"] or float(record["label_score_space"]) != row["score"]:
            raise ValueError(f"Trace text/label differs from training CSV at row {row_index}")
        for key in ("step_gain", "cumulative_delta"):
            if not math.isfinite(float(record[key])) or float(record[key]) <= 0:
                raise ValueError(f"Invalid positive score gain: {key}")
        if not record["before_text"] or not record["adversarial_text"]:
            raise ValueError("Empty trace text")
        if record["record_id"] in ids:
            raise ValueError(f"Duplicate trace: {record['record_id']}")
        ids.add(record["record_id"])
        group = grouped.setdefault(row_index, [])
        if group and group[0]["attack"] != attack:
            raise ValueError(f"Multiple attack assignments for essay {row_index}")
        group.append(record)
    counts = {attack: 0 for attack in RHI_ATTACKS}
    for group in grouped.values():
        counts[group[0]["attack"]] += 1
    if (not all(counts.values()) or counts["rudimentary"] != counts["hotflip"]
            or counts["injection_external"] != counts["injection_self_dup"]
            or counts["rudimentary"] != counts["injection_external"] * 2):
        raise ValueError(f"Unbalanced essay exposure: {counts}")
    return grouped, counts
