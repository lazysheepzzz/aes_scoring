"""Remote smoke B0 -> PAER: native global score, visible edit targets, 1024-token backward. No updates."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT/"outputs/smoke_longformer_asap2_b0_seed42/best")
    args = p.parse_args()
    import torch
    from paer.aes_rh_trainer import RHTraceCollator
    from paer.modeling_paer_v3 import PAERV3ForEssayScoring
    from paer.train_longformer_aes_baseline import read_fold
    from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU memory check requires remote CUDA")
    scorer = AESScorer(args.checkpoint, device="cuda")
    if scorer.model.config.model_type != "longformer":
        raise ValueError("Not a Longformer checkpoint")
    net = PAERV3ForEssayScoring(scorer.model).cuda().eval()
    tok = scorer.tokenizer
    rows = sorted(read_fold(ROOT/"data/train_fold0.csv"), key=lambda r: -len(r[1]))
    batch = tok([r[1] for r in rows[:2]], truncation=True, padding=True, max_length=1024,
                return_tensors="pt").to("cuda")
    if batch.input_ids.shape != (2, 1024):
        raise ValueError("Expected a 2 x 1024 memory probe")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ref = scorer.model(**batch).logits
        torch.testing.assert_close(net(**batch).global_logits, ref, rtol=1e-5, atol=1e-5)
    # Only a tokenizer wiring fixture, never saved as an essay or used in training.
    coll = RHTraceCollator(tok,1024,1,.1)
    t = rows[0][1]
    item = dict(original_text=t, before_text=t, adversarial_text="Example: "+t,
                has_adversarial=True, step_gain=.1, cumulative_delta=.1,
                label_score_space=rows[0][2], attack="injection_external")
    encoded = coll([item])
    if not encoded["cumulative_uplift_targets"].any():
        raise RuntimeError("Visible edit lost during token alignment")
    net.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = net(**batch).logits.float().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(net.token_evidence_head.weight.grad).all():
        raise RuntimeError("Non-finite Longformer backward")
    print(f"PASS: native global logits; visible edit; batch 2 x 1024 backward; "
          f"peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          "(probe only, excludes optimizer state). No weights updated.")


if __name__ == "__main__":
    main()
