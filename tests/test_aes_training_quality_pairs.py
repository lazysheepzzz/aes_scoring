"""Disposable fixtures only: no real essays, API calls, GPU or paid ledger writes."""
import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from paer import training_quality_pairs as t


class TrainingPairPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train = self.root / 'train.csv'
        self.valid = self.root / 'valid.csv'
        self.pool = self.root / 'pool'
        self.pool.mkdir()
        self.traces = self.pool / 'traces.jsonl'
        self.output = self.root / 'preview'
        self.ledger = self.root / 'old_paid_audit/budget_ledger.jsonl'
        attacks = ['rudimentary'] * 4 + ['hotflip'] * 4 + ['injection_external'] * 2 + ['injection_self_dup'] * 2
        self.rows, self.records = [], []
        for i, attack in enumerate(attacks):
            original = f'Fixture essay number {i}. Students discuss learning and books.'
            self.rows.append({'essay_id': f'train_{i}', 'full_text': original,
                              'score': i % 3 + 1, 'prompt_name': f'topic_{i % 2}'})
            before = original
            for n in (1, 2):
                after = before + f' Existing recorded sentence {n}.'
                self.records.append({'record_version': 1, 'record_id': f'{i}:{attack}:{n}',
                    'essay_id': f'train_{i}', 'row_index': i, 'attack': attack,
                    'label_score_space': i % 3 + 1, 'original_text': original,
                    'before_text': before, 'adversarial_text': after,
                    'victim_score_before': 2 + (n - 1) * .04,
                    'victim_score_after': 2 + n * .04, 'step_gain': .04,
                    'cumulative_delta': n * .04, 'accepted_edit_order': n})
                before = after
        self.write_csv(self.train, self.rows)
        self.write_csv(self.valid, [{'essay_id': 'valid_1', 'full_text': 'Completely distinct held out fixture.',
                                     'score': 3, 'prompt_name': 'valid_topic'}])
        self.write_pool()
        self.args = argparse.Namespace(train_csv=self.train, valid_csv=self.valid,
            trace_jsonl=self.traces, output_dir=self.output, budget_ledger=self.ledger,
            essays=6, seed=42, max_tokens=768, dry_run=False)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write_csv(path, rows):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['essay_id', 'full_text', 'score', 'prompt_name'])
            writer.writeheader()
            writer.writerows(rows)

    def write_pool(self):
        self.traces.write_text(''.join(json.dumps(r) + '\n' for r in self.records), encoding='utf-8')
        self.write_json(self.traces.with_suffix('.manifest.json'), {
            'mlm_excluded': True, 'source_hashes': {'train': t.file_hash(self.train)},
            'trace_sha256': t.file_hash(self.traces)})

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()), patch('socket.create_connection',
                side_effect=AssertionError('NO NETWORK')):
            return t.prepare(self.args)

    def test_exact_existing_steps_balanced_and_blind(self):
        self.prepare()
        manifest, values = t.verify_bundle(self.output)
        private, blind = values['private_pair_records.json'], values['blind_pairs.json']
        self.assertEqual(Counter(p['attack'] for p in private),
                         dict(rudimentary=2, hotflip=2, injection_external=1, injection_self_dup=1))
        self.assertEqual(len({p['parent_essay_id'] for p in private}), 6)
        by_id = {r['record_id']: r for r in self.records}
        for source, pair in zip(private, blind):
            old = by_id[source['record_id']]
            self.assertEqual(pair['text_A'], old['before_text'])
            self.assertEqual(pair['text_B'], old['adversarial_text'])
            self.assertTrue(source['original_human_score_does_not_label_step_A_or_B'])
            request = t.build_request(pair, 'forward', 768)
            data = json.loads(request['messages'][1]['content'])
            self.assertEqual(set(data), {'A', 'B', 'observed_edit_spans_A_to_B', 'context_scope'})
            self.assertNotIn('train_', request['messages'][1]['content'])
            for forbidden in ('recorded_step_gain', 'attack', 'original_human_score', 'row_index', 'parent_essay_id'):
                self.assertNotIn(forbidden, data)
        self.assertFalse(manifest['quality_labels_present'])
        self.assertFalse(manifest['text_generation'])
        self.assertEqual(manifest['network_calls'], 0)

    def test_first_six_are_also_family_balanced(self):
        self.args.essays = 12
        self.prepare()
        sources = t.read_json(self.output / 'private_pair_records.json')
        for start in (0, 6):
            self.assertEqual(Counter(p['attack'] for p in sources[start:start + 6]),
                             dict(rudimentary=2, hotflip=2, injection_external=1, injection_self_dup=1))

    def test_dry_run_writes_nothing(self):
        self.args.dry_run = True
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        preview = self.prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.ledger.parent.exists())
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        self.assertEqual(preview['labels_created'], 0)
        self.assertIsNone(preview['budget_snapshot']['used_or_reserved_cny'])

    def test_deterministic_selection_and_output_fingerprints(self):
        self.prepare()
        first, _ = t.verify_bundle(self.output)
        self.args.output_dir = self.root / 'second_preview'
        self.prepare()
        second, _ = t.verify_bundle(self.args.output_dir)
        self.assertEqual(first, second)

    def test_seed_changes_selection_not_scores(self):
        rows, groups, _ = t.load_groups(self.train, self.valid, self.traces)
        first = [p[0]['record_id'] for p in t.choose_records(rows, groups, 6, 42)]
        second = [p[0]['record_id'] for p in t.choose_records(rows, groups, 6, 43)]
        self.assertNotEqual(first, second)

    def test_no_overwrite_even_empty_directory(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_second_run_preserves_first(self):
        self.prepare()
        original = (self.output / 'source_manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(original, (self.output / 'source_manifest.json').read_bytes())

    def test_missing_pool_does_not_generate_it(self):
        self.args.trace_jsonl = self.root / 'missing/no_traces.jsonl'
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.args.trace_jsonl.parent.exists())
        self.assertFalse(self.output.exists())

    def test_source_hash_tampering_rejected(self):
        self.traces.write_text(self.traces.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'trace hash'):
            self.prepare()

    def test_wrong_train_hash_rejected(self):
        manifest = t.read_json(self.traces.with_suffix('.manifest.json'))
        manifest['source_hashes']['train'] = 'bad'
        self.write_json(self.traces.with_suffix('.manifest.json'), manifest)
        with self.assertRaisesRegex(ValueError, 'training CSV hash'):
            self.prepare()

    def test_mlm_records_and_manifest_rejected(self):
        self.records[0]['attack'] = 'mlm_guided'
        self.write_pool()
        with self.assertRaises(ValueError):
            self.prepare()
        self.records[0]['attack'] = 'rudimentary'
        self.write_pool()
        manifest = t.read_json(self.traces.with_suffix('.manifest.json'))
        manifest['mlm_excluded'] = False
        self.write_json(self.traces.with_suffix('.manifest.json'), manifest)
        with self.assertRaisesRegex(ValueError, 'exclude MLM'):
            self.prepare()

    def test_validation_parent_id_overlap_rejected(self):
        self.write_csv(self.valid, [{'essay_id': 'train_0', 'full_text': 'Other fixture.',
                                    'score': 1, 'prompt_name': ''}])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.prepare()

    def test_validation_normalized_text_overlap_rejected(self):
        self.write_csv(self.valid, [{'essay_id': 'valid_1', 'full_text': '  ' + self.rows[0]['full_text'] + '  ',
                                    'score': 1, 'prompt_name': ''}])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.prepare()

    def test_validation_intermediate_text_overlap_rejected(self):
        self.write_csv(self.valid, [{'essay_id': 'valid_1', 'full_text': self.records[0]['adversarial_text'],
                                    'score': 1, 'prompt_name': ''}])
        with self.assertRaisesRegex(ValueError, 'step text overlaps'):
            self.prepare()

    def test_score_metadata_inconsistency_rejected(self):
        self.records[0]['victim_score_after'] = 9
        self.write_pool()
        with self.assertRaisesRegex(ValueError, 'step_gain disagree'):
            self.prepare()

    def test_signed_cumulative_delta_is_preserved(self):
        for record in self.records:
            if record['attack'] == 'hotflip':
                record['cumulative_delta'] = -.1
        self.write_pool()
        self.prepare()
        private = t.read_json(self.output / 'private_pair_records.json')
        self.assertTrue(all(p['recorded_cumulative_delta_from_original'] == -.1
                            for p in private if p['attack'] == 'hotflip'))

    def test_not_enough_essays_fails_not_oversamples(self):
        self.args.essays = 18
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_quota_and_token_limit(self):
        for count in (0, 5, -6):
            self.args.essays = count
            with self.assertRaises(ValueError):
                self.prepare()
        self.args.essays, self.args.max_tokens = 6, 16
        with self.assertRaises(ValueError):
            self.prepare()

    def test_export_tampering_rejected(self):
        self.prepare()
        pairs = t.read_json(self.output / 'blind_pairs.json')
        pairs[0]['text_A'] += ' changed'
        self.write_json(self.output / 'blind_pairs.json', pairs)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            t.verify_bundle(self.output)

    def test_reverse_request_recomputes_exact_spans(self):
        pair = {'text_A': 'A good day.', 'text_B': 'A brilliant day.'}
        forward = json.loads(t.build_request(pair, 'forward', 768)['messages'][1]['content'])
        reverse = json.loads(t.build_request(pair, 'reverse', 768)['messages'][1]['content'])
        self.assertEqual(forward['A'], reverse['B'])
        self.assertEqual(forward['B'], reverse['A'])
        self.assertEqual(reverse['observed_edit_spans_A_to_B'], t.changed_spans(pair['text_B'], pair['text_A']))

    def test_exact_spans_reconstruct_without_rewriting(self):
        examples = [('A good day.', 'A brilliant day.'), ('one two', 'one new two'),
                    ('one two three', 'one three'), ('  text\n', ' text \n'),
                    ('Mixed CASE and tyop.', 'Mixed CASE and typo.'), ('x', 'xy'),
                    ('same', 'same'), ('甲乙 word.', '甲丙 word.')]
        for a, b in examples:
            pieces, cursor = [], 0
            for span in t.changed_spans(a, b):
                self.assertEqual(span['text_A'], a[span['a_start']:span['a_end']])
                self.assertEqual(span['text_B'], b[span['b_start']:span['b_end']])
                pieces.extend([a[cursor:span['a_start']], span['text_B']])
                cursor = span['a_end']
            pieces.append(a[cursor:])
            self.assertEqual(''.join(pieces), b)

    def test_single_relation_protocol_does_not_force_global_grade_change(self):
        self.assertIn('ONE quality relation', t.SYSTEM_PROMPT)
        self.assertIn('need NOT change the overall essay grade', t.SYSTEM_PROMPT)
        self.assertIn('null relation', t.SYSTEM_PROMPT)
        self.assertNotIn('local_relation', t.SYSTEM_PROMPT)
        self.assertNotIn('overall_relation', t.SYSTEM_PROMPT)

    def test_budget_shared_read_only_and_bounds(self):
        from paer.quality_relation_audit import BudgetLedger
        ledger = BudgetLedger(self.ledger, 30)
        ledger.reserve('old_request', Decimal('3'))
        ledger.settle('old_request', Decimal('2.310256'))
        ledger.reserve('interrupted_request', Decimal('.2'))
        before = self.ledger.read_bytes()
        preview = self.prepare()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertEqual(Decimal(preview['budget_snapshot']['used_or_reserved_cny']), Decimal('2.510256'))
        self.assertEqual(preview['base_order_requests'], 12)
        self.assertEqual(preview['max_requests_with_third_votes_and_two_attempts'], 36)
        self.assertGreater(Decimal(preview['maximum_protocol_reservation_bound_cny']),
                           Decimal(preview['base_order_reservation_bound_cny']))

    def test_budget_tampering_and_empty_ledger_rejected(self):
        self.ledger.parent.mkdir()
        self.ledger.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.prepare()
        self.ledger.write_text(json.dumps({'event_sha256': 'bad', 'previous': '0' * 64}) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash chain'):
            self.prepare()

    def test_refuses_old_audit_or_pool_child_output(self):
        for target in (self.ledger.parent, self.ledger.parent / 'child', self.pool / 'child'):
            self.args.output_dir = target
            with self.assertRaises(ValueError):
                self.prepare()


if __name__ == '__main__':
    unittest.main()
