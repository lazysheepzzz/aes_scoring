"""Tiny random CPU models/temporary fixtures only. No downloads or real training data."""
from __future__ import annotations

import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, BertTokenizerFast

from paer.expert_routing.model import AESLowRankExpertScorer, LowRankExpertLinear
from paer.expert_routing.prototypes import RoutingPrototypeBank
from paer.expert_routing.trainer import ExpertTrainer, ExpertTrainingConfig, ExpertTraceCollator
from paer.expert_routing.launch import (EXPERT_ONLY, EXPERT_PAER, parser_for, prepare_config, read_rows, smoke_indices)
from paer.rhi_experiment_utils import checkpoint_identity, save_json, sha256, validate_trace_groups
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack

torch.set_num_threads(1)


def tiny_base():
    config = DebertaV2Config(vocab_size=100, hidden_size=12, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=3, num_labels=1, max_position_embeddings=128,
        relative_attention=True, share_att_key=True, pos_att_type=['p2c', 'c2p'],
        max_relative_positions=16, hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
        pooler_dropout=0.)
    return DebertaV2ForSequenceClassification(config)


class ExpertModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.ids = torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]])
        self.mask = torch.ones_like(self.ids)

    def model(self, use_paer=True):
        return AESLowRankExpertScorer(tiny_base(), use_paer=use_paer, expert_rank=2).eval()

    def test_zero_residual_preserves_original_global_score(self):
        base = tiny_base().eval()
        with torch.no_grad():
            reference = base(input_ids=self.ids, attention_mask=self.mask).logits
            model = AESLowRankExpertScorer(base, use_paer=False, expert_rank=2).eval()
            result = model(input_ids=self.ids, attention_mask=self.mask)
        torch.testing.assert_close(result.logits, reference, rtol=0, atol=0)
        self.assertEqual(len(result.expert_gates), 4)
        self.assertTrue(all('query_proj' not in path for path in model.expert_paths))
        for gate in result.expert_gates:
            torch.testing.assert_close(gate.sum(-1), torch.ones_like(gate[..., 0]))

    def test_relative_attention_and_input_gradients_use_full_expert_path(self):
        model = self.model()
        with torch.no_grad():
            for adapter in model.adapters():
                for up in adapter.up:
                    up.weight.normal_(std=.2)
        embeds = model.get_input_embeddings()(self.ids).detach().requires_grad_(True)
        result = model(inputs_embeds=embeds, attention_mask=self.mask)
        result.logits.sum().backward()
        self.assertTrue(torch.isfinite(embeds.grad).all())
        self.assertGreater(float(embeds.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for adapter in model.adapters() for p in adapter.local_gate.parameters()))

    def test_experts_can_change_global_scoring_not_just_token_correction(self):
        model = self.model(False)
        with torch.no_grad():
            before = model(input_ids=self.ids, attention_mask=self.mask).global_logits.clone()
            for adapter in model.adapters():
                for up in adapter.up:
                    up.weight.normal_(std=1.)
            after = model(input_ids=self.ids, attention_mask=self.mask).global_logits
        self.assertGreater(float((before - after).abs().max()), 0)

    def test_padding_does_not_change_global_gate_or_score(self):
        model = self.model()
        with torch.no_grad():
            for adapter in model.adapters():
                for up in adapter.up:
                    up.weight.normal_(std=.2)
            short = model(input_ids=self.ids[:1], attention_mask=self.mask[:1])
            long = model(input_ids=torch.cat([self.ids[:1], torch.tensor([[9, 10, 11]])], 1),
                         attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0, 0]]))
        torch.testing.assert_close(short.logits, long.logits, rtol=1e-4, atol=1e-6)
        for a, b in zip(short.expert_gates, long.expert_gates):
            torch.testing.assert_close(a, b[:, :4], rtol=1e-4, atol=1e-6)

    def test_paer_still_only_subtracts_nonnegative_correction(self):
        model = self.model()
        result = model(input_ids=self.ids, attention_mask=self.mask)
        self.assertTrue((result.correction >= 0).all())
        self.assertTrue((result.logits <= result.base_logits + 1e-7).all())

    def test_checkpoint_roundtrip_both_modes(self):
        for enabled in (False, True):
            with self.subTest(use_paer=enabled), tempfile.TemporaryDirectory() as root:
                model = self.model(enabled)
                with torch.no_grad():
                    for adapter in model.adapters():
                        for up in adapter.up:
                            up.weight.normal_(std=.1)
                    before = model(input_ids=self.ids, attention_mask=self.mask).logits.clone()
                model.save_pretrained(root)
                loaded = AESLowRankExpertScorer.from_pretrained(root).eval()
                with torch.no_grad():
                    after = loaded(input_ids=self.ids, attention_mask=self.mask).logits.clone()
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertEqual(loaded.use_paer, enabled)
                del loaded, model
                gc.collect()

    def test_bfloat16_checkpoint_path_is_dtype_consistent(self):
        model = AESLowRankExpertScorer(tiny_base().to(torch.bfloat16), expert_rank=2).eval()
        with torch.no_grad():
            result = model(input_ids=self.ids, attention_mask=self.mask)
        self.assertTrue(torch.isfinite(result.logits).all())

    def test_no_hidden_checkpoint_recomputation_or_context_fallback(self):
        model = self.model()
        model.base_model.deberta.encoder.gradient_checkpointing = True
        with self.assertRaisesRegex(RuntimeError, 'checkpointing'):
            model(input_ids=self.ids, attention_mask=self.mask)
        adapter = LowRankExpertLinear(torch.nn.Linear(4, 4), rank=2, experts=3)
        with self.assertRaisesRegex(RuntimeError, 'full AES'):
            adapter(torch.randn(1, 4, 4))

    def test_optimizer_has_every_trainable_parameter_once(self):
        for enabled in (False, True):
            model = self.model(enabled)
            trainer = ExpertTrainer.__new__(ExpertTrainer)
            trainer.model = model
            trainer.config = SimpleNamespace(expert_learning_rate=1e-4, paer_head_learning_rate=1e-4,
                                             learning_rate=1e-5, weight_decay=.01)
            groups = trainer.optimizer_groups()
            ids = [id(p) for group in groups for p in group['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
            expert_ids = {id(p) for p in model.expert_parameters()}
            self.assertTrue(all(group['lr'] == 1e-4 for group in groups
                                if any(id(p) in expert_ids for p in group['params'])))


class PrototypeTests(unittest.TestCase):
    def test_single_family_microbatches_learn_against_detached_history(self):
        bank = RoutingPrototypeBank(2, 3)
        for family in (0, 1, 2, 0):
            raw = [torch.randn(1, 4, 3, requires_grad=True) for _ in range(2)]
            loss, metrics = bank.loss_and_update(tuple(x.softmax(-1) for x in raw),
                                                 torch.tensor([[0., 1., 1., 0.]]), torch.tensor([family]))
            loss.backward()
            self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in raw))
        self.assertEqual(bank.counts.tolist(), [2, 1, 1])
        self.assertGreater(metrics['router_loss'], 0.)
        self.assertIsNone(bank.prototypes.grad_fn)
        self.assertFalse(bank.prototypes.requires_grad)

    def test_no_visible_edits_do_not_fabricate_supervision(self):
        bank = RoutingPrototypeBank(1, 3)
        gate = torch.randn(1, 4, 3, requires_grad=True).softmax(-1)
        loss, metrics = bank.loss_and_update((gate,), torch.zeros(1, 4), torch.tensor([0]))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(metrics['router_supervised_essays'], 0)
        self.assertEqual(bank.counts.sum().item(), 0)

    def test_long_edits_get_one_essay_vote_not_more_token_votes(self):
        bank = RoutingPrototypeBank(1, 3)
        weights = torch.tensor([[[.8, .1, .1]]]).repeat(1, 20, 1).requires_grad_(True)
        bank.loss_and_update((weights,), torch.ones(1, 20), torch.tensor([2]))
        torch.testing.assert_close(bank.prototypes[0, 2], torch.tensor([.8, .1, .1]))
        self.assertEqual(bank.counts.tolist(), [0, 0, 1])

    def test_invalid_family_ids_are_not_new_experts(self):
        bank = RoutingPrototypeBank(1, 3)
        with self.assertRaises(ValueError):
            bank.loss_and_update((torch.ones(1, 2, 3) / 3,), torch.ones(1, 2), torch.tensor([3]))


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.train, self.valid = self.data / 'train.csv', self.data / 'valid.csv'
        self.train.write_text('essay_id,score,full_text\n' + ''.join(f't{i},{2+i%3},Essay {i} text\n' for i in range(60)), encoding='utf-8')
        self.valid.write_text('essay_id,score,full_text\nv1,3,Validation text\nv2,4,Other validation text\n', encoding='utf-8')
        self.ckpt = self.root / 'b0'
        tiny_base().save_pretrained(self.ckpt)
        vocab = self.root / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'Essay', 'text', 'edited',
                                    'Validation', 'Other', 'validation'] + [str(i) for i in range(60)]) + '\n', encoding='utf-8')
        # Transformers 5 renamed vocab_file to vocab; retain compatibility with 4.x.
        tokenizer = BertTokenizerFast(vocab=str(vocab), vocab_file=str(vocab), do_lower_case=False)
        tokenizer.save_pretrained(self.ckpt)
        self.trace = self.root / 'pool/traces.jsonl'
        self.trace.parent.mkdir()
        attacks = ['rudimentary', 'rudimentary', 'hotflip', 'hotflip', 'injection_external', 'injection_self_dup']
        records = [{'record_id': f'record{i}', 'row_index': i, 'essay_id': f't{i}',
            'attack': attack, 'original_text': f'Essay {i} text', 'before_text': f'Essay {i} text',
            'adversarial_text': f'Essay {i} text edited', 'label_score_space': 2+i%3,
            'step_gain': .05, 'cumulative_delta': .05, 'accepted_edit_order': 1} for i, attack in enumerate(attacks)]
        self.trace.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        save_json(self.trace.with_suffix('.manifest.json'), {'mlm_excluded': True,
            'source_hashes': {'train': sha256(self.train)}, 'trace_sha256': sha256(self.trace),
            'checkpoint': checkpoint_identity(self.ckpt), 'arguments': {'max_length': 1024}})

    def tearDown(self):
        gc.collect()
        self.tmp.cleanup()

    def args(self, experiment=EXPERT_PAER, extra=()):
        parser, backend = parser_for(experiment)
        args = parser.parse_args(['--checkpoint-path', str(self.ckpt), '--train-csv', str(self.train),
            '--valid-csv', str(self.valid), '--trace-jsonl', str(self.trace), '--output-dir', str(self.root / experiment),
            '--smoke', '--max-train-samples', '12', '--max-valid-samples', '2', '--expert-rank', '2',
            '--precision', 'float32', '--no-progress', *extra])
        return args, backend

    def test_smoke_has_all_peer_families_and_deterministic_existing_rows(self):
        args, backend = self.args()
        config, inputs = prepare_config(args, EXPERT_PAER, backend)
        self.assertEqual(inputs['essay_counts_by_attack'], {'rudimentary': 2, 'hotflip': 2,
                                                           'injection_external': 1, 'injection_self_dup': 1})
        self.assertEqual(inputs['n_clean_rows'], 12)
        self.assertEqual(config['num_epochs'], 1)
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(inputs['quality_judge_used'])
        other, mode = self.args(EXPERT_ONLY)
        control, _ = prepare_config(other, EXPERT_ONLY, mode)
        self.assertEqual(control['selected_train_row_indices'], config['selected_train_row_indices'])
        self.assertNotEqual(control['training_mode'], config['training_mode'])

    def test_old_output_and_tampered_pool_are_rejected(self):
        args, backend = self.args()
        args.output_dir.mkdir()
        with self.assertRaises(FileExistsError):
            prepare_config(args, EXPERT_PAER, backend)
        args.output_dir = self.root / 'new_output'
        self.trace.write_text(self.trace.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pool'):
            prepare_config(args, EXPERT_PAER, backend)

    def test_partial_data_requires_smoke_and_no_prefix_trace_truncation(self):
        args, backend = self.args()
        args.smoke = False
        with self.assertRaisesRegex(ValueError, 'Partial'):
            prepare_config(args, EXPERT_PAER, backend)
        args.smoke, args.max_trace_records = True, 5
        with self.assertRaisesRegex(ValueError, 'Prefix'):
            prepare_config(args, EXPERT_PAER, backend)

    def test_invalid_configuration_is_rejected_before_output_creation(self):
        for name, value in (('max_train_samples', 0), ('learning_rate', float('nan')),
                            ('expert_layers', 3), ('per_device_train_batch_size', 0)):
            with self.subTest(name=name):
                args, backend = self.args()
                setattr(args, name, value)
                with self.assertRaises(ValueError):
                    prepare_config(args, EXPERT_PAER, backend)
                self.assertFalse(args.output_dir.exists())

    def test_collator_retains_edit_mask_with_negative_cumulative_gain(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.ckpt)
        collator = ExpertTraceCollator(tokenizer, 32, 1., .1)
        examples = [{'original_text': 'Essay 1 text', 'before_text': 'Essay 1 text',
            'adversarial_text': 'Essay 1 text edited', 'label_score_space': 3,
            'step_gain': .05, 'cumulative_delta': -.01, 'attack': name, 'has_adversarial': True}
            for name in ('injection_external', 'injection_self_dup')]
        batch = collator(examples)
        self.assertEqual(batch['expert_families'].tolist(), [2, 2])
        self.assertEqual(float(batch['cumulative_uplift_targets'].sum()), 0.)
        self.assertGreater(float(batch['expert_edit_mask'].sum()), 0.)

    def test_changed_implementation_is_rejected_before_model_loading(self):
        args, backend = self.args()
        config, inputs = prepare_config(args, EXPERT_PAER, backend)
        inputs['source_sha256']['paer/expert_routing/model.py'] = 'changed'
        args.output_dir.mkdir()
        save_json(args.output_dir / 'rhi_training_inputs.json', inputs)
        with patch.object(AESLowRankExpertScorer, 'from_base_checkpoint',
                          side_effect=AssertionError('must not load a model')):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                ExpertTrainer(ExpertTrainingConfig(**config))

    def test_tiny_cpu_training_both_variants_and_scorer_reload(self):
        for experiment in (EXPERT_ONLY, EXPERT_PAER):
            args, backend = self.args(experiment)
            config, inputs = prepare_config(args, experiment, backend)
            config['gradient_accumulation_steps'] = 2
            args.output_dir.mkdir()
            save_json(args.output_dir / 'rhi_training_inputs.json', inputs)
            trainer = ExpertTrainer(ExpertTrainingConfig(**config))
            trainer.train()
            self.assertTrue((args.output_dir / 'best/paer_experts.pt').is_file())
            resource = json.loads((args.output_dir / 'expert_resource_metrics.json').read_text())
            self.assertTrue(all(v > 0 for v in resource['prototype_family_observations']))
            scorer = AESScorer(args.output_dir / 'best', device='cpu', dtype=torch.float32)
            with torch.no_grad():
                values = scorer.score_batch(['Essay 1 text', 'Essay 1 text edited'], batch_size=2)
            self.assertEqual(len(values), 2)
            self.assertTrue(all(torch.isfinite(torch.tensor(values))))
            attacker = HotFlipAttack(scorer, n_steps=1, n_sample_pos=2, top_k_per_pos=1,
                max_candidates_per_step=2, batch_size=2, max_token_edit_rate=None)
            adversarial_text, history = attacker.attack('Essay 1 text edited')
            self.assertIsInstance(adversarial_text, str)
            self.assertTrue(torch.isfinite(torch.tensor(scorer.score_single(adversarial_text))))
            del attacker, scorer, trainer
            gc.collect()


if __name__ == '__main__':
    unittest.main()
