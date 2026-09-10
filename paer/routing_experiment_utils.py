"""Read-only provenance checks for post-freeze PAER routing experiments."""
from pathlib import Path

from paer.rhi_experiment_utils import ROOT, checkpoint_identity, read_json, sha256

ATTACK_FILES = {
    "rudimentary": "rudimentary/rudimentary_details.json",
    "hotflip": "hotflip/hotflip_details.json",
    "injection_external": "injection_family/injection_external_details.json",
    "injection_self_dup": "injection_family/injection_self_dup_details.json",
}
DEFAULT_EVALUATION = ROOT / "outputs/aes_rhi_evaluation_development_seed42"


def frozen_reference(directory):
    """Require the actual evaluated PAER weights/code, not a mutable best alias."""
    directory = Path(directory).resolve()
    protocol = read_json(directory / "rhi_run_binding.json")
    if protocol["protocol"] != "frozen_checkpoint_evaluation_v1" or protocol["suite"] != "rhi":
        raise ValueError("Expected a frozen RHI evaluation")
    checkpoint = Path(protocol["checkpoints"]["paer"]["path"])
    if checkpoint_identity(checkpoint) != protocol["checkpoints"]["paer"]["files"]:
        raise ValueError("PAER weights/config differ from frozen evaluation")
    if read_json(checkpoint / "paer_config.json")["model_type"] != "paer_aes_v3":
        raise ValueError("This diagnostic requires PAER-v3 signed token aggregation")
    if sha256(protocol["data"]) != protocol["data_sha256"]:
        raise ValueError("Frozen evaluation data changed")
    for name, digest in protocol["code_hashes"].items():
        if sha256(ROOT / name.replace("\\", "/")) != digest:
            raise ValueError(f"Frozen evaluation code changed: {name}")
    bank = ROOT / "injection/wikipedia_sentences_100.txt"
    if sha256(bank) != protocol["bank_sha256"]:
        raise ValueError("Frozen Injection sentence bank changed")
    for family in ("rudimentary", "hotflip", "injection_family"):
        result_dir = directory / "paer" / family
        hashes = read_json(result_dir / "completed_result_hashes.json")
        required = {"run_manifest.json", "clean_qwk.json", "asr_summary.json"}
        required.update(Path(p).name for p in ATTACK_FILES.values() if Path(p).parent.name == family)
        if family == "injection_family":
            required.add("injection_family_summary.json")
        if not required.issubset(hashes):
            raise ValueError(f"Incomplete frozen result markers: {family}")
        for name, digest in hashes.items():
            if sha256(result_dir / name) != digest:
                raise ValueError(f"Frozen result changed: {result_dir / name}")
    return protocol, checkpoint


def equal_family_average(rows, fields):
    """R, H, and the mean of Injection subattacks each receive one third."""
    by_attack = {row["attack"]: row for row in rows}
    if len(by_attack) != len(rows) or set(by_attack) != set(ATTACK_FILES):
        raise ValueError("Require exactly R, H, Injection External and Self-Dup")
    if len({row["n_pairs"] for row in rows}) != 1:
        raise ValueError("Unequal paired sample counts")
    return {key: (by_attack["rudimentary"][key] + by_attack["hotflip"][key]
                  + (by_attack["injection_external"][key] + by_attack["injection_self_dup"][key]) / 2) / 3
            for key in fields}


def code_identity(*names):
    return {name: sha256(ROOT / name) for name in names}
