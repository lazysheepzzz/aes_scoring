"""Disposable fixtures and fake scorers only; no ML/GPU/API/real essay access."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import test_aes_training_quality_pairs as fixtures
import test_aes_training_quality_judge as judge_fixtures
from paer import evaluate_aes_fixed_quality_pairs as e
from paer import quality_relation_audit as q
from paer import training_quality_pairs as t
from paer import training_quality_judge as j


class FixedQualityPairTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingPairPreparationTests()
        self.f.setUp()
        self.output_tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.output_tmp.name) / 'new_output'
        ledger = q.BudgetLedger(self.f.ledger, 30)
        ledger.reserve('old_request', 3)
        ledger.settle('old_request', Decimal('2.310256'))
        self.f.prepare()
        self.judge_dir = self.f.root / 'judge'
        judge_args = argparse.Namespace(input_dir=self.f.output, run_dir=self.judge_dir,
            output_dir=self.f.root / 'judge_report', max_pairs=0, timeout=90,
            allow_upload=True, summarize_only=False, dry_run=False)
        def transport(body, *unused):
            data = json.loads(body['messages'][1]['content'])
            label = 'worsened' if len(data['B']) > len(data['A']) else 'improved'
            return judge_fixtures.response(body, label)
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ,
                {'DEEPSEEK_API_KEY': 'fixture_not_real'}):
            j.run(judge_args, transport)
        self.paths = {}
        for model in e.MODELS:
            path = self.f.root / 'checkpoints' / model
            path.mkdir(parents=True)
            q.save_json(path / 'config.json', {'num_labels': 1})
            (path / 'model.safetensors').write_bytes(b'fixture_not_real_weights_' + model.encode())
            self.paths[model] = path
        q.save_json(self.paths['paer'] / 'paer_config.json', {'model_type': 'paer_aes_v3'})
        (self.paths['paer'] / 'paer_heads.pt').write_bytes(b'fixture_not_real_heads')
        self.selections = {}
        for model, step in (('mixed', 1000), ('paer', 400)):
            selection = self.f.root / f'{model}_uniform30' / 'best_checkpoint.json'
            q.save_json(selection, {'checkpoint_name': f'gstep{step}', 'checkpoint_path': str(self.paths[model]),
                                    'max_checkpoint_step': 1400})
            q.save_json(selection.parent / 'rhi_run_binding.json', {
                'selection_attacks': ['rudimentary', 'hotflip', 'injection_family'],
                'mlm_used_for_selection': False,
                'arguments': {'selection_steps': 30, 'rudimentary_selection_steps': 30,
                              'injection_selection_steps': 30},
                'candidate_weights': {f'gstep{step}': e.checkpoint_files(self.paths[model])}})
            self.selections[model] = selection
        self.args = argparse.Namespace(input_dir=self.f.output, judge_dir=self.judge_dir, out=self.output,
            b0_checkpoint=self.paths['b0'], mixed_selection=self.selections['mixed'],
            paer_selection=self.selections['paer'], batch_size=2, max_length=1024,
            dtype='float32', device='cpu', numeric_epsilon=1e-5, no_progress=True, dry_run=False)
        self.calls = []

    def tearDown(self):
        self.output_tmp.cleanup()
        self.f.tearDown()

    def backend(self, model, path, texts, args):
        self.calls.append((model, tuple(texts)))
        scale = {'b0': .01, 'mixed': .005, 'paer': .002}[model]
        return {text: {'score': 2. + len(text) * scale, 'input_sha256': t.digest(text),
                       'encoded_length': min(len(text), args.max_length)} for text in texts}

    def evaluate(self, backend=None):
        with contextlib.redirect_stdout(io.StringIO()), patch('socket.create_connection',
                side_effect=AssertionError('NO NETWORK')):
            return e.run(self.args, backend or self.backend)

    def loaded_rows(self):
        return e.load_audited_pairs(self.f.output, self.judge_dir)[1]

    def test_same_texts_sequential_models_and_no_old_artifact_writes(self):
        ledger = self.f.ledger.read_bytes()
        before = {str(p): p.read_bytes() for p in self.f.output.iterdir() if p.is_file()}
        summary = self.evaluate()
        self.assertEqual([c[0] for c in self.calls], list(e.MODELS))
        self.assertTrue(all(c[1] == self.calls[0][1] for c in self.calls))
        self.assertEqual(summary['n_stable_worsened'], 6)
        self.assertEqual(ledger, self.f.ledger.read_bytes())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.f.output.iterdir() if p.is_file()})
        self.assertEqual(summary['preservation_of_legitimate_gains'], 'NOT_EVALUABLE_NO_STABLE_IMPROVED_PAIRS')
        self.assertLess(summary['paired_comparisons']['stable_worsened_all']['mean_paired_paer_minus_mixed_delta'], 0)
        cache = (self.output / 'b0_fixed_text_scores.json').read_text(encoding='utf-8')
        self.assertNotIn('Fixture essay', cache)
        self.assertTrue((self.output / 'fixed_quality_pair_summary.md').is_file())
        self.assertNotIn('qwk', summary)

    def test_resume_reuses_same_bound_scores(self):
        self.evaluate()
        self.evaluate()
        self.assertEqual(len(self.calls), 3)

    def test_dry_run_no_model_loads_or_file_writes(self):
        self.args.dry_run = True
        ledger = self.f.ledger.read_bytes()
        result = self.evaluate(lambda *unused: self.fail('Dry run loaded model'))
        self.assertFalse(self.output.exists())
        self.assertEqual(self.f.ledger.read_bytes(), ledger)
        self.assertEqual(result['n_stable_worsened'], 6)

    def test_rejects_old_and_unbound_outputs(self):
        self.output.mkdir()
        with self.assertRaisesRegex(FileExistsError, 'unbound'):
            self.evaluate()
        for output in (self.f.output / 'child', self.judge_dir / 'child', self.f.ledger.parent / 'child',
                       self.paths['paer'] / 'child'):
            self.args.out = output
            with self.assertRaisesRegex(ValueError, 'outside'):
                self.evaluate()
        self.assertEqual(self.calls, [])

    def test_setting_change_cannot_reuse_old_scores(self):
        self.evaluate()
        self.args.batch_size = 1
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.evaluate()

    def test_vote_hash_or_prepared_prompt_tampering_stops_before_scoring(self):
        path = next((self.judge_dir / 'votes').glob('*.json'))
        vote = q.read_json(path)
        vote['judgment']['normalized_relation'] = 'improved'
        q.save_json(path, vote)
        with self.assertRaisesRegex(ValueError, 'Cached judgment changed'):
            self.evaluate()
        self.assertEqual(self.calls, [])

    def test_judge_code_binding_tampering_rejected(self):
        path = self.judge_dir / 'judge_binding.json'
        binding = q.read_json(path)
        binding['runner_code_sha256'] = 'changed'
        q.save_json(path, binding)
        with self.assertRaisesRegex(ValueError, 'Judge binding differs'):
            self.evaluate()

    def test_cache_tampering_rejected(self):
        self.evaluate()
        path = self.output / 'mixed_fixed_text_scores.json'
        cache = q.read_json(path)
        cache['scores'][0]['value']['score'] += 1
        q.save_json(path, cache)
        with self.assertRaisesRegex(ValueError, 'cache binding/content changed'):
            self.evaluate()

    def test_different_tokenized_inputs_rejected(self):
        def different(model, path, texts, args):
            scores = self.backend(model, path, texts, args)
            if model == 'mixed':
                scores[texts[0]]['input_sha256'] = t.digest('different tokenizer result')
            return scores
        with self.assertRaisesRegex(ValueError, 'different tokenized inputs'):
            self.evaluate(different)
        self.assertFalse((self.output / 'fixed_quality_pair_results.json').exists())

    def test_invisible_edit_retained_in_all_and_excluded_from_common_visible(self):
        row = self.loaded_rows()[0]
        def invisible(model, path, texts, args):
            scores = self.backend(model, path, texts, args)
            scores[row['text_B']] = dict(scores[row['text_A']])
            return scores
        result = self.evaluate(invisible)
        self.assertEqual(result['cohorts']['stable_worsened_all']['n_pairs'], 6)
        self.assertEqual(result['cohorts']['stable_worsened_common_model_visible']['n_pairs'], 5)
        self.assertEqual(result['cohorts']['stable_worsened_all']['models']['paer']['pooled_descriptive']['identical_model_input_pairs'], 1)

    def test_bad_scores_stop_not_silent_nan_results(self):
        def bad(model, path, texts, args):
            scores = self.backend(model, path, texts, args)
            scores[texts[0]]['score'] = float('nan')
            return scores
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.evaluate(bad)

    def test_checkpoint_changed_during_scoring_rejected(self):
        def mutate(model, path, texts, args):
            scores = self.backend(model, path, texts, args)
            (path / 'model.safetensors').write_bytes(b'different fixture weights')
            return scores
        with self.assertRaisesRegex(ValueError, 'Checkpoint changed'):
            self.evaluate(mutate)

    def test_failed_second_model_resumes_without_reloading_first(self):
        def fail(model, path, texts, args):
            if model == 'mixed':
                raise RuntimeError('Fixture interruption')
            return self.backend(model, path, texts, args)
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.evaluate(fail)
        self.assertTrue((self.output / 'b0_fixed_text_scores.json').is_file())
        self.evaluate()
        self.assertEqual([c[0] for c in self.calls], ['b0', 'mixed', 'paer'])

    def test_invalid_options(self):
        self.args.max_length = 2048
        with self.assertRaises(ValueError):
            self.evaluate()
        self.args.max_length, self.args.batch_size = 1024, 0
        with self.assertRaises(ValueError):
            self.evaluate()
        self.args.batch_size, self.args.numeric_epsilon = 2, float('nan')
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_not_the_uniform_selection_is_rejected(self):
        selection = q.read_json(self.selections['mixed'])
        selection.pop('max_checkpoint_step')
        q.save_json(self.selections['mixed'], selection)
        with self.assertRaisesRegex(ValueError, 'uniform-30 selection'):
            self.evaluate()

    def test_ten_step_hotflip_binding_rejected_even_if_filename_says_uniform30(self):
        path = self.selections['mixed'].parent / 'rhi_run_binding.json'
        value = q.read_json(path)
        value['arguments']['selection_steps'] = 10
        q.save_json(path, value)
        with self.assertRaisesRegex(ValueError, 'not uniform-30'):
            self.evaluate()

    def test_checkpoint_changed_since_selection_rejected(self):
        (self.paths['mixed'] / 'model.safetensors').write_bytes(b'changed after selection')
        with self.assertRaisesRegex(ValueError, 'selection-time identity'):
            self.evaluate()

    def test_constant_offset_does_not_change_pair_delta(self):
        rows = self.loaded_rows()
        texts = list(dict.fromkeys(v for r in rows for v in (r['text_A'], r['text_B'])))
        scores = {m: {text: {'score': len(text) * .01 + (-2 if m == 'paer' else 0),
                            'input_sha256': t.digest(text), 'encoded_length': len(text)} for text in texts} for m in e.MODELS}
        result, _ = e.summarize_rows(rows, scores, 1e-5)
        difference = result['paired_comparisons']['stable_worsened_all']['mean_paired_paer_minus_mixed_delta']
        self.assertAlmostEqual(difference, 0)

    def test_equal_family_macro_not_four_attack_or_pooled_average(self):
        rows = self.loaded_rows()
        removed_h = False
        for row in rows:
            if row['source']['attack'] == 'hotflip' and not removed_h:
                row['decision'] = {'status': 'abstained_uncertain_or_mixed', 'relation': None}
                removed_h = True
        texts = list(dict.fromkeys(v for r in rows for v in (r['text_A'], r['text_B'])))
        scores = {m: {text: {'score': 1., 'input_sha256': t.digest(text), 'encoded_length': len(text)}
                      for text in texts} for m in e.MODELS}
        for row in rows:
            if row['source']['attack'] == 'rudimentary':
                for model in e.MODELS:
                    scores[model][row['text_B']]['score'] = 2.
        result, _ = e.summarize_rows(rows, scores, 1e-5)
        primary = result['cohorts']['stable_worsened_all']['models']['b0']
        self.assertEqual(primary['pooled_descriptive']['n_pairs'], 5)
        self.assertAlmostEqual(primary['pooled_descriptive']['increase_rate_gt_numeric_epsilon'], .4)
        self.assertAlmostEqual(primary['rhi_equal_family_macro']['increase_rate_gt_numeric_epsilon'], 1 / 3)

    def test_empty_cohort_is_null_not_perfect_performance(self):
        result = e.metrics([], [], 1e-5)
        self.assertEqual(result['n_pairs'], 0)
        self.assertIsNone(result['mean_delta'])
        self.assertIsNone(result['increase_rate_ge_0_1'])

    def test_majority_improved_not_promoted_and_edited_report_ignored(self):
        row = self.loaded_rows()[0]
        binding = q.read_json(self.judge_dir / 'judge_binding.json')
        for order in ('forward', 'forward_repeat'):
            body = t.build_request(row, order, binding['max_tokens'])
            content = judge_fixtures.response(body, 'improved')['choices'][0]['message']['content']
            vote = {'pair_id': row['pair_id'], 'order': order, 'status': 'ok',
                    'judge_binding_sha256': q.fingerprint(binding), 'request_sha256': t.digest(body),
                    'content': content, 'judgment': j.parse_judgment(content, row, order)}
            vote['record_sha256'] = q.fingerprint(vote)
            q.save_json(self.judge_dir / 'votes' / f"{row['pair_id']}_{order}.json", vote)
        q.save_json(self.f.root / 'judge_report/pair_decisions.json', [{'relation': 'improved'}])
        result = self.evaluate()
        self.assertEqual(result['n_prepared_pairs'], 6)
        self.assertEqual(result['n_stable_worsened'], 5)
        self.assertEqual(result['cohorts']['stable_improved']['n_pairs'], 0)
        self.assertEqual(result['cohorts']['other_or_unresolved']['n_pairs'], 1)
        details = q.read_json(self.output / 'fixed_quality_pair_details.json')
        changed = next(item for item in details if item['pair_id'] == row['pair_id'])
        self.assertEqual(changed['decision']['status'], 'majority_after_conflict')
        self.assertFalse(changed['primary_stable_worsened'])

    def test_judge_lock_prevents_scoring_while_votes_are_updated(self):
        with q.RunLock(self.judge_dir):
            with self.assertRaisesRegex(RuntimeError, 'Another audit process'):
                self.evaluate()
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
