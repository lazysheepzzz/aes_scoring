"""Mock transport only. No API key/real essays/GPU/real budget ledger access."""
import argparse
import contextlib
import io
import json
import os
import unittest
from decimal import Decimal
from urllib.error import HTTPError
from unittest.mock import patch

import test_aes_training_quality_pairs as fixtures
from paer import quality_relation_audit as q
from paer import training_quality_pairs as t
from paer import training_quality_judge as j


def response(body, label='unchanged', status='judged', evidence=True, finish='stop', usage=True):
    data = json.loads(body['messages'][1]['content'])
    value = {'decision_status': status, 'relation_B_to_A': label if status == 'judged' else None,
             'effect_strength': 'none' if label == 'unchanged' else 'subtle',
             'affected_dimensions': ['clarity_meaning'], 'reason': 'The supplied edit does not clarify the argument.',
             'evidence_A': data['A'][:25] if evidence else 'NOT PRESENT IN ESSAY',
             'evidence_B': data['B'][:25]}
    return {'id': 'fixture_response', 'model': 'fixture-model-alias',
            'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 100} if usage else {}}


class TrainingQualityJudgeTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingPairPreparationTests()
        self.f.setUp()
        ledger = q.BudgetLedger(self.f.ledger, 30)
        ledger.reserve('original_audit', Decimal('3'))
        ledger.settle('original_audit', Decimal('2.310256'))
        self.f.prepare()
        self.args = argparse.Namespace(input_dir=self.f.output, run_dir=self.f.root / 'judge',
            output_dir=self.f.root / 'report', max_pairs=6, timeout=90,
            dry_run=False, allow_upload=True, summarize_only=False)
        self.calls = []

    def tearDown(self):
        self.f.tearDown()

    def transport(self, body, key, timeout):
        self.calls.append(body)
        self.assertEqual(key, 'unit_test_placeholder_not_a_real_key')
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertEqual(body['messages'][0]['content'], t.SYSTEM_PROMPT)
        self.assertEqual(set(json.loads(body['messages'][1]['content'])),
                         {'A', 'B', 'observed_edit_spans_A_to_B', 'context_scope'})
        return response(body)

    def run_judge(self, transport=None):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ,
                {'DEEPSEEK_API_KEY': 'unit_test_placeholder_not_a_real_key'}), \
                patch('socket.create_connection', side_effect=AssertionError('NO REAL NETWORK')):
            return j.run(self.args, transport or self.transport)

    def test_pilot_resume_reuses_votes_and_original_ledger(self):
        original_manifest = (self.f.output / 'source_manifest.json').read_bytes()
        before = self.f.ledger.read_bytes()
        self.args.max_pairs = 2
        self.assertEqual(self.run_judge(), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(self.f.ledger.read_bytes().startswith(before))
        self.args.max_pairs = 0
        self.assertEqual(self.run_judge(), 0)
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(self.run_judge(), 0)
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(original_manifest, (self.f.output / 'source_manifest.json').read_bytes())
        self.assertFalse((self.f.output / 'votes').exists())
        self.assertFalse((self.f.ledger.parent / 'budget_status.json').exists())
        report = t.read_json(self.args.output_dir / 'training_edit_quality_results.json')
        self.assertEqual(report['resolved'], 6)
        self.assertEqual(sum(r['unchanged'] for r in report['rows']), 6)
        self.assertFalse(report['training_supervision_approved'])
        self.assertGreater(q.BudgetLedger(self.f.ledger, 30).spent(), Decimal('2.310256'))

    def test_dry_run_no_key_no_files_no_budget_writes(self):
        self.args.dry_run = True
        before = {str(p): p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()}
        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(j.run(self.args), 0)
        after = {str(p): p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.args.run_dir.exists())

    def test_upload_needs_authorization_and_key(self):
        self.args.allow_upload = False
        with self.assertRaisesRegex(ValueError, 'allow-upload'):
            self.run_judge()
        self.args.allow_upload = True
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'DEEPSEEK_API_KEY'):
            j.run(self.args, self.transport)
        self.assertFalse(self.args.run_dir.exists())

    def test_forward_reverse_normalization(self):
        pair = {'text_A': 'An old sentence.', 'text_B': 'A revised sentence.'}
        body = t.build_request(pair, 'reverse', 768)
        parsed = j.parse_judgment(response(body, 'improved')['choices'][0]['message']['content'], pair, 'reverse')
        self.assertEqual(parsed['normalized_relation'], 'worsened')

    def test_one_pair_label_conflict_triggers_only_one_third_vote(self):
        self.args.max_pairs = 1
        calls = []
        def conflict(body, *unused):
            calls.append(body)
            # Equal raw labels in opposite directions conflict after normalization.
            return response(body, 'improved')
        self.run_judge(conflict)
        self.assertEqual(len(calls), 3)
        decisions = t.read_json(self.args.output_dir / 'pair_decisions.json')
        self.assertEqual(decisions[0]['status'], 'majority_after_conflict')
        self.assertEqual(decisions[0]['relation'], 'improved')
        self.assertTrue(decisions[0]['order_conflict'])
        row = t.read_json(self.args.output_dir / 'training_edit_quality_results.json')['rows'][0]
        self.assertEqual(row['stable_only_counts']['improved'], 0)

    def test_uncertain_and_mixed_stay_unknown_without_extra_vote(self):
        self.args.max_pairs = 1
        calls = []
        def abstain(body, *unused):
            calls.append(body)
            return response(body, status='uncertain' if len(calls) == 1 else 'mixed')
        self.run_judge(abstain)
        self.assertEqual(len(calls), 2)
        first = t.read_json(self.args.output_dir / 'pair_decisions.json')[0]
        self.assertIsNone(first['relation'])
        self.assertEqual(first['status'], 'abstained_uncertain_or_mixed')

    def test_quotation_error_retries_bounded_and_never_unchanged(self):
        self.args.max_pairs = 1
        calls = []
        def invalid(body, *unused):
            calls.append(body)
            return response(body, evidence=False)
        self.run_judge(invalid)
        self.assertEqual(len(calls), 4)
        self.run_judge(invalid)
        self.assertEqual(len(calls), 4)
        first = t.read_json(self.args.output_dir / 'pair_decisions.json')[0]
        self.assertIsNone(first['relation'])
        self.assertEqual(first['status'], 'pending_or_failed')
        self.assertEqual(len(list((self.args.run_dir / 'attempts').glob('*.json'))), 4)

    def test_missing_usage_keeps_full_reservations(self):
        self.args.max_pairs = 1
        initial = q.BudgetLedger(self.f.ledger, 30).spent()
        self.run_judge(lambda body, *unused: response(body, usage=False))
        ledger = q.BudgetLedger(self.f.ledger, 30)
        self.assertEqual(len([e for e in ledger.events if e['kind'] == 'reserve']), 5)
        self.assertEqual(len([e for e in ledger.events if e['kind'] == 'settle']), 1)
        self.assertGreater(ledger.spent(), initial)

    def test_incomplete_completion_is_not_a_label(self):
        self.args.max_pairs = 1
        self.run_judge(lambda body, *unused: response(body, finish='length'))
        self.assertIsNone(t.read_json(self.args.output_dir / 'pair_decisions.json')[0]['relation'])

    def test_three_different_labels_remain_unresolved(self):
        def vote(label):
            return {'status': 'ok', 'judgment': {'normalized_relation': label}}
        result = j.final_decision({'forward': vote('improved'), 'reverse': vote('worsened'),
                                   'forward_repeat': vote('unchanged')})
        self.assertIsNone(result['relation'])
        self.assertEqual(result['status'], 'unresolved_conflict')

    def test_response_usage_overrun_stops_before_second_request(self):
        calls = []
        def overrun(body, *unused):
            calls.append(1)
            value = response(body)
            value['usage']['prompt_tokens'] = 100000
            return value
        self.assertEqual(self.run_judge(overrun), 2)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(t.read_json(self.args.output_dir / 'pair_decisions.json')[0]['relation'])

    def test_original_budget_lock_blocks_concurrent_audit(self):
        with q.RunLock(self.f.ledger.parent):
            with self.assertRaisesRegex(RuntimeError, 'Another audit process'):
                self.run_judge()
        self.assertEqual(self.calls, [])

    def test_auth_failure_stops_after_one_request_and_writes_report(self):
        self.args.max_pairs = 1
        calls = []
        def fatal(*unused):
            calls.append(1)
            raise HTTPError('https://api.deepseek.com/chat/completions', 401, 'Unauthorized', {}, None)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
            self.run_judge(fatal)
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.args.run_dir / 'budget_status.json').is_file())
        self.assertTrue((self.args.output_dir / 'pair_decisions.json').is_file())

    def test_budget_exhaustion_never_calls_transport(self):
        ledger = q.BudgetLedger(self.f.ledger, 30)
        ledger.reserve('other_budget_use', Decimal('27.689744'))
        self.assertEqual(self.run_judge(), 2)
        self.assertEqual(self.calls, [])

    def test_missing_or_replaced_ledger_refuses_budget_reset(self):
        saved = self.f.ledger.read_bytes()
        self.f.ledger.unlink()
        with self.assertRaisesRegex(ValueError, 'ledger missing'):
            self.run_judge()
        ledger = q.BudgetLedger(self.f.ledger, 30)
        ledger.reserve('fresh_unrelated_ledger', 0)
        with self.assertRaisesRegex(ValueError, 'replaced/truncated'):
            self.run_judge()
        self.assertNotEqual(saved, self.f.ledger.read_bytes())
        self.assertEqual(self.calls, [])

    def test_unknown_ledger_snapshot_cannot_enable_paid_calls(self):
        self.f.args.output_dir = self.f.root / 'unknown_preparation'
        self.f.args.budget_ledger = self.f.root / 'missing_paid_audit/budget_ledger.jsonl'
        self.f.prepare()
        self.args.input_dir = self.f.args.output_dir
        with self.assertRaisesRegex(ValueError, 'unknown prior spend'):
            self.run_judge()

    def test_unknown_interrupted_request_is_not_resent_for_free(self):
        self.args.max_pairs = 1
        _, _, _, values, _, binding = j.load_run_inputs(self.args)
        pair = values['blind_pairs.json'][0]
        request_id = t.digest({'binding': binding, 'pair_id': pair['pair_id'], 'order': 'forward'})
        q.BudgetLedger(self.f.ledger, 30).reserve(f'training-edit:{request_id}:0', Decimal('.1'))
        self.run_judge()
        self.assertEqual(len(self.calls), 2)  # forward attempt 1 + reverse attempt 0
        vote = t.read_json(self.args.run_dir / 'votes' / f"{pair['pair_id']}_forward.json")
        self.assertTrue(vote['attempt'].endswith(':1'))

    def test_changed_cache_rejected(self):
        self.args.max_pairs = 1
        self.run_judge()
        path = next((self.args.run_dir / 'votes').glob('*.json'))
        vote = t.read_json(path)
        vote['judgment']['normalized_relation'] = 'improved'
        path.write_text(json.dumps(vote), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Cached judgment changed'):
            self.run_judge()

    def test_different_run_directory_does_not_reset_attempts(self):
        self.args.max_pairs = 1
        self.run_judge()
        self.assertEqual(len(self.calls), 2)
        self.args.run_dir = self.f.root / 'different_cache'
        self.args.output_dir = self.f.root / 'different_report'
        self.run_judge()
        # Missing cache may consume the second, already bounded attempt, not a new allowance.
        self.assertEqual(len(self.calls), 4)
        self.args.run_dir = self.f.root / 'third_cache'
        self.args.output_dir = self.f.root / 'third_report'
        self.run_judge()
        self.assertEqual(len(self.calls), 4)

    def test_summarize_only_never_calls_transport(self):
        self.run_judge()
        count = len(self.calls)
        self.args.summarize_only = True
        self.args.allow_upload = False
        self.run_judge(lambda *args: self.fail('Summary attempted an API request'))
        self.assertEqual(len(self.calls), count)

    def test_refuses_unbound_directories_and_protected_locations(self):
        self.args.run_dir.mkdir()
        with self.assertRaisesRegex(FileExistsError, 'unbound run'):
            self.run_judge()
        self.args.run_dir = self.f.ledger.parent / 'new_child'
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_judge()
        self.args.run_dir = self.f.root / 'safe_run'
        self.args.output_dir = self.f.output / 'new_child'
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_judge()
        self.args.output_dir = self.f.root / 'foreign_report'
        self.args.output_dir.mkdir()
        with self.assertRaisesRegex(FileExistsError, 'unbound report'):
            self.run_judge()
        self.assertEqual(self.calls, [])

    def test_invalid_options_and_prepared_prompt_tampering(self):
        for value in (-1, 7):
            self.args.max_pairs = value
            with self.assertRaisesRegex(ValueError, 'max-pairs'):
                self.run_judge()
        self.args.max_pairs = 6
        self.args.timeout = float('nan')
        with self.assertRaisesRegex(ValueError, 'Timeout'):
            self.run_judge()
        self.args.timeout = 90
        protocol_path = self.f.output / 'review_protocol.json'
        protocol = t.read_json(protocol_path)
        protocol['system_prompt'] = 'Changed prompt'
        protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.run_judge()

    def test_parse_rejects_unknown_as_unchanged_and_wrong_schema(self):
        pair = {'text_A': 'A fixture original text.', 'text_B': 'A fixture modified text.'}
        body = t.build_request(pair, 'forward', 768)
        value = json.loads(response(body)['choices'][0]['message']['content'])
        value['decision_status'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'null'):
            j.parse_judgment(json.dumps(value), pair, 'forward')
        value['relation_B_to_A'] = None
        self.assertIsNone(j.parse_judgment(json.dumps(value), pair, 'forward')['normalized_relation'])
        value['confidence'] = 'high'
        with self.assertRaisesRegex(ValueError, 'fields'):
            j.parse_judgment(json.dumps(value), pair, 'forward')


if __name__ == '__main__':
    unittest.main()
