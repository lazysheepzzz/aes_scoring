"""Run the original clean-continuation trainer with recoverable single-GPU checkpoints."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import checkpoint_identity, read_json, save_json, sha256

OUT = ROOT / "outputs/longformer_asap2_c0_resumable_seed42"
BASE = ROOT / "outputs/longformer_asap2_b0_seed42/best"


@contextmanager
def exclusive_run(directory):
    """OS releases this advisory lock on crash/reboot; an old lock file is harmless."""
    with (directory/"writer.lock").open("a+b") as stream:
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class C0Checkpoints:
    def __init__(self, directory, signature, resume):
        self.directory, self.signature, self.resume = directory, signature, resume

    def path(self, step):
        return self.directory / f"resume-step-{step:06d}.pt"

    def save(self, trainer, **cursor):
        import numpy as np
        import torch
        dst = self.path(cursor["step"])
        # Only complete .pt files are eligible for resume. Previous steps remain intact.
        if dst.exists():
            raise FileExistsError(dst)
        rng = np.random.get_state()
        state = dict(**cursor, signature=self.signature, model=trainer.scorer.model.state_dict(),
                     optimizer=trainer.optimizer.state_dict(), scheduler=trainer.scheduler.state_dict(),
                     python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else [])
        tmp = dst.with_suffix(".partial")
        with tmp.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(dst)
        print(f"[C0 checkpoint] {dst.name} (full optimizer step)", flush=True)

    def load(self, step):
        import torch
        state = torch.load(self.path(step), map_location="cpu", weights_only=True, mmap=True)
        if state["signature"] != self.signature or state["step"] != step:
            raise ValueError("Checkpoint does not belong to this run")
        return state

    def restore(self, trainer):
        if not self.resume:
            return None
        steps = [int(p.stem.split("-")[-1]) for p in self.directory.glob("resume-step-*.pt")]
        if not steps:
            print("[C0 resume] No committed optimizer checkpoint yet; restarting from bound B0.", flush=True)
            return None
        state = self.load(max(steps))
        trainer.scorer.model.load_state_dict(state.pop("model"))
        trainer.optimizer.load_state_dict(state.pop("optimizer"))
        trainer.scheduler.load_state_dict(state.pop("scheduler"))
        if state["best_step"] is not None and not self.path(state["best_step"]).is_file():
            raise FileNotFoundError("Best snapshot required for final export is missing")
        print(f"[C0 resume] optimizer step={state['step']}, epoch={state['epoch']+1}, "
              f"consumed epoch batches={state['epoch_steps']}", flush=True)
        return state

    @staticmethod
    def restore_rng(state):
        import numpy as np
        import torch
        random.setstate(state["python_rng"])
        name, keys, *extra = state["numpy_rng"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *extra))
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def export(self, trainer, tag, step):
        dst = self.directory/tag
        # A reboot between final and best export may leave one fully published export.
        provenance = {"step": step, "signature": self.signature}
        if dst.exists():
            if read_json(dst/"c0_export.json") != provenance:
                raise ValueError(f"Foreign export: {dst}")
            return
        temp = Path(tempfile.mkdtemp(prefix=f".{tag}-", dir=self.directory))
        trainer.scorer.model.save_pretrained(temp)
        trainer.tokenizer.save_pretrained(temp)
        save_json(temp/"c0_export.json", provenance)
        temp.rename(dst)

    def finish(self, trainer, step, best_step, best_qwk, metrics):
        import math
        if not math.isfinite(metrics["qwk"]):
            raise ValueError("Final QWK is not finite")
        self.export(trainer, "final", step)
        if best_step is not None and metrics["qwk"] <= best_qwk:
            trainer.scorer.model.load_state_dict(self.load(best_step)["model"])
        else:
            best_step, best_qwk = step, metrics["qwk"]
        self.export(trainer, "best", best_step)
        save_json(self.directory/"completed.json", dict(final_step=step, final_metrics=metrics,
                  best_step=best_step, best_qwk=best_qwk, signature=self.signature))
        print(f"[DONE] Longformer C0: best step={best_step}, QWK={best_qwk:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    resume = parser.parse_args().resume
    from whitebox.aes_stage2_training_launcher import (
        CLEAN_CONTINUATION, _configure_environment, build_config, build_parser, validate_config,
    )
    args = build_parser(CLEAN_CONTINUATION).parse_args([
        "--checkpoint-path", str(BASE), "--output-dir", str(OUT),
        "--train-csv", str(ROOT/"data/train_fold0.csv"), "--valid-csv", str(ROOT/"data/valid_fold0.csv"),
        "--per-device-train-batch-size", "2", "--gradient-accumulation-steps", "16",
        "--per-device-eval-batch-size", "2"])
    _configure_environment(args)
    import torch
    import transformers
    from text_scoring_adv_training.training.aes_trainer import AESAdversarialConfig, AESAdversarialTrainer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use the remote CUDA bf16 training environment")
    config = build_config(args, CLEAN_CONTINUATION)
    validate_config(config)
    if config["eval_every"] != config["save_every"]:
        raise ValueError("All candidate best weights must be checkpointed")
    protocol = dict(config=config, b0=checkpoint_identity(BASE), train=sha256(args.train_csv),
                    valid=sha256(args.valid_csv), torch=str(torch.__version__), transformers=transformers.__version__,
                    cuda_devices=torch.cuda.device_count(), format_version=1)
    if not resume:
        OUT.mkdir(parents=True, exist_ok=False)
    with exclusive_run(OUT):
        binding = OUT/"c0_binding.json"
        if resume:
            if read_json(binding) != protocol:
                raise ValueError("B0, data, environment or training configuration changed")
            if (OUT/"completed.json").exists():
                print(f"[COMPLETE] {OUT}")
                return
        else:
            save_json(binding, protocol)
            save_json(OUT/"launcher_config.json", config)
        trainer = AESAdversarialTrainer(AESAdversarialConfig(**config))
        trainer.train(checkpoints=C0Checkpoints(OUT, sha256(binding), resume))


if __name__ == "__main__":
    main()
