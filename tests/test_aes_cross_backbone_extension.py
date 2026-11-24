import unittest

import torch
from transformers import LongformerConfig, LongformerForSequenceClassification

from paer.modeling_paer_v3 import PAERV3ForEssayScoring
from paer.prepare_original_asap_folds import RANGES, prepare, target
from paer.evaluate_original_asap_prompt_qwk import evaluate
from paer.run_longformer_rhi_stage import BASE, C0, SEL, commands
from paper_results.summarize_aes_upward_band_crossings import count
from text_scoring_adv_training.evaluation.aes.scorer import _longformer_cls_global_mask


class CrossBackbone(unittest.TestCase):
    def test_native_global_logits_and_embeddings(self):
        cfg = LongformerConfig(vocab_size=80, hidden_size=16, num_attention_heads=2,
                               num_hidden_layers=1, intermediate_size=32, attention_window=[4],
                               max_position_embeddings=44, pad_token_id=1, num_labels=1)
        base = LongformerForSequenceClassification(cfg).eval()
        base.register_forward_pre_hook(_longformer_cls_global_mask, with_kwargs=True)
        model = PAERV3ForEssayScoring(base).eval()
        ids = torch.tensor([[0,7,10,2,1,1],[0,17,15,20,11,2]])
        mask = (ids != 1).long()
        native = base(input_ids=ids, attention_mask=mask).logits
        emb = base.get_input_embeddings()(ids).detach().requires_grad_(True)
        routed = model(inputs_embeds=emb, attention_mask=mask)
        for actual in (model(input_ids=ids, attention_mask=mask).global_logits,
                       base(inputs_embeds=emb,attention_mask=mask).logits, routed.global_logits):
            torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-5)
        routed.logits.sum().backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertTrue(routed.attention_weights[0,4:].eq(0).all())

    def test_internal_grades(self):
        row = dict(original_score=.5,perturbed_score=1.5,original_band=0,perturbed_band=2)
        self.assertEqual(count([row],None),1)
        with self.assertRaises(ValueError):
            count([row],[.5,1.5,2.5,3.5,4.5])

    def test_longformer_paths(self):
        for mode in ("mixed","paer"):
            self.assertIn(BASE, commands(mode)[0])
            self.assertIn(C0/"best", commands("select-"+mode)[0])
            self.assertIn(SEL[mode]/"best_checkpoint.json", commands("evaluate-pair")[0])

    def test_prompt_scales_and_partition(self):
        rows = []
        for k,(lo,hi) in RANGES.items():
            self.assertEqual(target(k,lo),0)
            self.assertEqual(target(k,hi),5)
            for j in range(20):
                rows.append(dict(essay_id=f"{k}-{j}",essay_set=k,essay=f"p{k} text{j//2}",
                                 domain1_score=lo if j%2 else hi))
        split = prepare(rows,42)
        texts = [{r["full_text"] for r in group} for group in split.values()]
        self.assertEqual(sum(map(len,texts)), len(set.union(*texts)))
        self.assertEqual(sum(map(len,split.values())),len(rows))
        self.assertEqual(split,prepare(rows,42))
        preds = {r["essay_id"]:r["model_target_0_5"] for r in split["test"]}
        self.assertEqual(evaluate(split["test"], preds), {p:1. for p in RANGES})
