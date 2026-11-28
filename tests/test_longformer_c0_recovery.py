import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from paer.train_resumable_longformer_c0 import C0Checkpoints
from text_scoring_adv_training.training.aes_trainer import AESAdversarialTrainer


class Net(torch.nn.Sequential):
    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory)/"weights.pt")


class Tokenizer:
    def save_pretrained(self, directory):
        (Path(directory)/"tokenizer.json").write_text("{}")


class Tiny(AESAdversarialTrainer):
    def __init__(self, directory):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.config = SimpleNamespace(num_epochs=3, per_device_train_batch_size=2, seed=42,
                                      gradient_accumulation_steps=3, eval_every=2, save_every=2,
                                      output_dir=str(directory), show_progress=False)
        self.train_dataset = list(range(9))  # 5 batches/epoch, accumulation spans epochs
        self.collator = lambda b: torch.tensor(b).float().view(-1,1)
        self.scorer = SimpleNamespace(model=Net(torch.nn.Dropout(.3),torch.nn.Linear(1,1)))
        self.optimizer = torch.optim.AdamW(self.scorer.model.parameters(), lr=.003)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.9)
        self.tokenizer = Tokenizer()
        self.order = []

    def train_step(self, b):
        self.order.append(b.flatten().tolist())
        loss = self.scorer.model(b).square().mean()*(.5+random.random()+np.random.random())
        (loss/3).backward()
        return {"total_loss":loss.item(),"base_loss":loss.item(),"hotflip_loss":0.}

    def evaluate(self):
        torch.rand(1)  # validation also consumes RNG
        return {"qwk": 1/self.scheduler.last_epoch, "mae": .5}


class PowerCut(C0Checkpoints):
    def save(self, trainer, **cursor):
        super().save(trainer, **cursor)
        if cursor["step"] == 2:
            raise KeyboardInterrupt


class RecoveryTests(unittest.TestCase):
    def test_restart_matches_original_across_epoch_boundary(self):
        with TemporaryDirectory() as temp:
            old, new = [Path(temp)/name for name in ("old","new")]
            new.mkdir()
            normal = Tiny(old)
            normal.train()
            part = Tiny(new)
            with self.assertRaises(KeyboardInterrupt):
                part.train(PowerCut(new, "same", False))
            # A partially written follow-up is not selected.
            (new/"resume-step-000006.partial").write_bytes(b"incomplete")
            fresh = Tiny(new)
            with self.assertRaisesRegex(ValueError, "belong"):
                C0Checkpoints(new, "different", True).restore(fresh)
            fresh.train(C0Checkpoints(new, "same", True))
            self.assertEqual(part.order+fresh.order, normal.order)
            self.assertEqual(fresh.scheduler.state_dict(), normal.scheduler.state_dict())
            for tag in ("best","final"):
                a,b=[torch.load(p/tag/"weights.pt", weights_only=True) for p in (old,new)]
                for k in a:
                    self.assertTrue(torch.equal(a[k],b[k]))
            state=json.loads((new/"completed.json").read_text())
            self.assertEqual((state["best_step"], state["final_step"]), (2,5))
            # CPU optimizer tensors can still reference the mmap; release before Windows cleanup.
            del fresh
            import gc
            gc.collect()
