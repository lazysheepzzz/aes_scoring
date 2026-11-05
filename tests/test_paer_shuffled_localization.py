"""CPU-only tests for the matched-count shuffled supervision control."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from paer.aes_rh_trainer import RHTraceCollator, RHTrainingConfig
from paer.localization_shuffle import shuffled_targets
from paer.shuffled_localization_preflight import check_shuffle


class ToyTokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id = 0

    def __call__(self, texts, *, max_length, **kwargs):
        rows = [[1, *[int(s) for s in t.split()][:max_length - 2], 2] for t in texts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        }


def trace(row, gain=0.07):
    return {
        "row_index": row,
        "original_text": "3 4 5 6 7 8 9 10 11",
        "before_text": "3 4 5 6 7 8 9 10 11",
        "adversarial_text": "3 4 99 6 7 8 9 10 11",
        "label_score_space": 3,
        "has_adversarial": True,
        "step_gain": 0.1,
        "cumulative_delta": gain,
        "attack": "rudimentary",
    }


class ShuffleTests(unittest.TestCase):
    def test_only_cumulative_targets_change_and_rng_is_untouched(self):
        items = [trace(i) for i in range(8)] + [trace(8, -0.1), trace(9, 0)]
        items.append({**trace(10), "original_text": "3 4", "before_text": "3 4",
                      "adversarial_text": "3 5"})
        original = RHTraceCollator(ToyTokenizer(), 32, 1.0, 0.1)
        shuffled = RHTraceCollator(ToyTokenizer(), 32, 1.0, 0.1, shuffled_localization_seed=42001)
        python_state, torch_state = random.getstate(), torch.random.get_rng_state()
        base, control = original(items), shuffled(items)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        key = "cumulative_uplift_targets"
        for name in base:
            if name != key:
                if isinstance(base[name], torch.Tensor):
                    self.assertTrue(torch.equal(base[name], control[name]), name)
                else:
                    self.assertEqual(base[name], control[name], name)
        self.assertFalse(torch.equal(base[key], control[key]))
        self.assertTrue(torch.equal(base[key].sort(dim=1).values, control[key].sort(dim=1).values))
        self.assertTrue((control[key][control["localization_mask"] == 0] == 0).all())
        self.assertTrue((control[key][8:10] == 0).all())
        self.assertTrue(torch.equal(control[key], shuffled(items[::-1])[key].flip(0)))

    def test_sampling_all_eligible_positions_is_not_forced_disjoint(self):
        self.assertEqual(shuffled_targets([0, 1, 0], [False, True, False], seed=1,
                                         row_index=0, adversarial_text="single"), [0, 1, 0])

    def test_shuffle_flag_cannot_be_silently_used_in_mixed_at(self):
        with self.assertRaisesRegex(ValueError, "requires PAER-v3"):
            RHTrainingConfig("mixed_at_rh", "b0", "train", "traces", "dev", "out",
                             shuffled_localization_seed=42001)

    def test_cpu_preflight_counts_only_states_reachable_during_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=ToyTokenizer()
        ) as tokenizer:
            root = Path(tmp)
            train, traces = root / "train.csv", root / "traces.jsonl"
            train.write_text("text,score\n" + ("3 4 5 6 7 8 9 10 11,3\n" * 8), encoding="utf-8")
            traces.write_text("\n".join(json.dumps(trace(i)) for i in range(8) for _ in range(2)),
                              encoding="utf-8")
            report = check_shuffle(dict(training_mode="paer_rh_v3", checkpoint_path="b0",
                                        train_csv=str(train), trace_jsonl=str(traces), valid_csv="dev",
                                        output_dir="out", num_epochs=1,
                                        shuffled_localization_seed=42001))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["by_attack"]["rudimentary"]["checked_states"], 8)
            self.assertGreater(report["by_attack"]["rudimentary"]["moved_states"], 0)
            self.assertTrue(tokenizer.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
