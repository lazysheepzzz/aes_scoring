"""Quotation-failure rejudging tests; no API, GPU, or real credentials."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paer import quality_relation_audit as q
from paer import retry_aes_quality_evidence_failures as repair


def judgment(label='worsened', a='Before.', b='After.'):
    return {'relation_B_to_A': label, 'confidence': 'high', 'affected_dimensions': ['organization'],
            'reason': 'The revised expression disrupts coherence.', 'evidence_A': a, 'evidence_B': b}


def response(value=None):
    return {'model': 'deepseek-flash', 'id': 'fake', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value or judgment())}}]}


class EvidenceRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.audit = self.root / 'audit'
        self.pair = {'original_text': 'Before.', 'modified_text': 'After.',
                     'context': {'task_prompt_text': None, 'source_material': None}}
        self.pair['pair_id'] = q.fingerprint(self.pair)
        self.records = [{'pair_id': self.pair['pair_id'], 'essay_id': '1', 'model': 'b0',
                         'attack': 'rudimentary', 'delta': 0.12}]
        manifest = {'pair_records_sha256': q.fingerprint(self.records),
                    'blind_pairs_sha256': q.fingerprint([self.pair]), 'sources': []}
        q.save_json(self.audit / 'source_manifest.json', manifest)
        q.save_json(self.audit / 'private_pair_records.json', self.records)
        q.save_json(self.audit / 'blind_pairs.json', [self.pair])
        self.binding = q.judge_binding(self.audit, 512)
        q.save_json(self.audit / 'judge_binding.json', self.binding)
        self.store_vote('forward', 'error', judgment(a='NOT IN TEXT'))
        self.store_vote('reverse', 'ok', judgment('improved', a='After.', b='Before.'))
        ledger = q.BudgetLedger(self.audit / 'budget_ledger.jsonl', 30)
        ledger.reserve('original_request', 0.03)
        ledger.settle('original_request', 0.001)
        self.args = argparse.Namespace(audit_dir=self.audit, repair_dir=None, output_dir=self.root / 'new_report',
                                       timeout=30, dry_run=False, allow_upload=True, summarize_only=False)

    def tearDown(self):
        self.tmp.cleanup()

    def store_vote(self, order, status, value):
        vote = {'pair_id': self.pair['pair_id'], 'order': order, 'status': status,
                'judge_binding_sha256': q.fingerprint(self.binding), 'content': json.dumps(value),
                'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}
        if status == 'ok':
            vote['judgment'] = q.parse_judgment(vote['content'], self.pair, order)
        else:
            vote['error_type'] = 'ValueError'
        vote['record_sha256'] = q.fingerprint(vote)
        q.save_json(self.audit / 'votes' / f"{self.pair['pair_id']}_{order}.json", vote)

    def run_repair(self, transport):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'FAKE'}), contextlib.redirect_stdout(io.StringIO()):
            repair.run(self.args, transport=transport)

    def test_only_failed_direction_and_preserved_original_votes(self):
        originals = {p: q.file_hash(p) for p in (self.audit / 'votes').glob('*.json')}
        calls = []
        def fake(body, *unused):
            calls.append(body)
            return response()
        self.run_repair(fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['messages'][0]['content'], q.SYSTEM_PROMPT + repair.QUOTATION_REMINDER)
        self.assertEqual(originals, {p: q.file_hash(p) for p in originals})
        self.assertEqual(q.BudgetLedger(self.audit / 'budget_ledger.jsonl', 30).spent(),
                         q.money(0.001) + q.usage_cost(response()['usage']))
        results = q.read_json(self.args.output_dir / 'quality_relation_results.json')
        self.assertEqual(results['unique_decision_status_counts'], {'order_consistent': 1})
        self.assertEqual(results['rows'][0]['worsened'], 1)

    def test_resume_does_not_rebill(self):
        calls = []
        def fake(*unused):
            calls.append(1)
            return response()
        self.run_repair(fake)
        spent = q.BudgetLedger(self.audit / 'budget_ledger.jsonl', 30).spent()
        self.run_repair(fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(q.BudgetLedger(self.audit / 'budget_ledger.jsonl', 30).spent(), spent)

    def test_dry_run_has_no_writes_or_network(self):
        self.args.dry_run = True
        before = q.file_hash(self.audit / 'budget_ledger.jsonl')
        self.run_repair(lambda *unused: self.fail('No network'))
        self.assertEqual(q.file_hash(self.audit / 'budget_ledger.jsonl'), before)
        self.assertFalse((self.audit / 'evidence_repair_run01').exists())

    def test_new_disagreement_gets_one_tiebreak_not_a_preferred_label(self):
        self.store_vote('reverse', 'ok', judgment('unchanged', a='After.', b='Before.'))
        calls = []
        def fake(*unused):
            calls.append(1)
            return response(judgment('improved'))
        self.run_repair(fake)
        self.assertEqual(len(calls), 2)
        results = q.read_json(self.args.output_dir / 'quality_relation_results.json')
        self.assertEqual(results['unique_decision_status_counts'], {'majority_after_conflict': 1})
        self.assertEqual(results['rows'][0]['improved'], 1)

    def test_original_failed_tiebreak_only_is_rejudged(self):
        self.store_vote('forward', 'ok', judgment('worsened'))
        self.store_vote('reverse', 'ok', judgment('worsened', a='After.', b='Before.'))
        self.store_vote('forward_repeat', 'error', judgment(a='FAKE QUOTE'))
        calls = []
        def fake(*unused):
            calls.append(1)
            return response()
        self.run_repair(fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(q.read_json(self.args.output_dir / 'quality_relation_results.json')
                         ['unique_decision_status_counts'], {'majority_after_conflict': 1})

    def test_invalid_quotes_never_become_ok_and_retries_are_bounded(self):
        calls = []
        def fake(*unused):
            calls.append(1)
            return response(judgment(a='FAKE QUOTE'))
        self.run_repair(fake)
        self.run_repair(fake)
        self.assertEqual(len(calls), 2)
        results = q.read_json(self.args.output_dir / 'quality_relation_results.json')
        self.assertEqual(results['rows'][0]['n_resolved'], 0)

    def test_budget_exhaustion_stops_before_api(self):
        ledger = q.BudgetLedger(self.audit / 'budget_ledger.jsonl', 30)
        ledger.reserve('other_unknown', 29.999)
        self.run_repair(lambda *unused: self.fail('No budget'))
        self.assertEqual(q.read_json(self.audit / 'budget_status.json')['remaining_cny'], '0')

    def test_no_new_budget_without_parent_ledger(self):
        (self.audit / 'budget_ledger.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'budget reset'):
            self.run_repair(lambda *unused: self.fail('No network'))

    def test_empty_evidence_is_not_a_repair(self):
        self.run_repair(lambda *unused: response(judgment(a='', b='')))
        results = q.read_json(self.args.output_dir / 'quality_relation_results.json')
        self.assertEqual(results['rows'][0]['n_resolved'], 0)

    def test_summary_only_no_key_or_api(self):
        self.run_repair(lambda *unused: response())
        self.args.summarize_only = True
        self.args.allow_upload = False
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}), contextlib.redirect_stdout(io.StringIO()):
            repair.run(self.args, transport=lambda *unused: self.fail('Summary only'))
