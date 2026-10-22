"""Budgeted edit-relation feasibility audit, separate from the old overall audit.

No training/attack imports. The already prepared texts and prompt remain immutable.
Only this explicit runner can upload them; summary and dry-run are offline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError

from paer import quality_relation_audit as q
from paer import training_quality_pairs as t

ROOT = t.ROOT
DEFAULT_INPUT = ROOT / 'artifacts/quality_assessment/training_edit_relations_seed42_preview01'
DEFAULT_RUN = ROOT / 'artifacts/quality_assessment/training_edit_relations_seed42_judge01'
DEFAULT_OUTPUT = ROOT / 'outputs/quality_assessment/training_edit_relations_seed42_judge01'
ORDERS = ('forward', 'reverse', 'forward_repeat')
LABELS = ('improved', 'unchanged', 'worsened')
DIMENSIONS = {'language_accuracy', 'clarity_meaning', 'coherence_redundancy'}


def parse_judgment(content, pair, order):
    value = json.loads(content)
    expected = {'decision_status', 'relation_B_to_A', 'effect_strength',
                'affected_dimensions', 'reason', 'evidence_A', 'evidence_B'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Unexpected judgment fields')
    status, label = value['decision_status'], value['relation_B_to_A']
    if status not in ('judged', 'uncertain', 'mixed'):
        raise ValueError('Invalid decision status')
    if (status == 'judged' and label not in LABELS) or (status != 'judged' and label is not None):
        raise ValueError('Uncertain/mixed is null, never an unchanged fallback')
    if value['effect_strength'] not in ('none', 'subtle', 'clear', 'uncertain'):
        raise ValueError('Invalid effect strength')
    dims = value['affected_dimensions']
    if not isinstance(dims, list) or any(not isinstance(d, str) or d not in DIMENSIONS for d in dims):
        raise ValueError('Invalid affected dimensions')
    for key in ('reason', 'evidence_A', 'evidence_B'):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 2000:
            raise ValueError(f'Invalid or empty {key}')
    if len(value['reason'].split()) > 80:
        raise ValueError('Reason exceeds the bound 80-word protocol')
    a, b = pair['text_A'], pair['text_B']
    if order == 'reverse':
        a, b = b, a
    if value['evidence_A'] not in a or value['evidence_B'] not in b:
        raise ValueError('Evidence not present in supplied essay')
    normalized = q.normalized_label(label, order) if label is not None else None
    return {**value, 'normalized_relation': normalized}


def final_decision(votes):
    primary = [votes.get(o) for o in ORDERS[:2]]
    if any(not v or v['status'] != 'ok' for v in primary):
        return {'status': 'pending_or_failed', 'relation': None, 'order_conflict': False}
    labels = [v['judgment']['normalized_relation'] for v in primary]
    if any(label is None for label in labels):
        return {'status': 'abstained_uncertain_or_mixed', 'relation': None, 'order_conflict': False}
    if labels[0] == labels[1]:
        return {'status': 'order_consistent', 'relation': labels[0], 'order_conflict': False}
    repeat = votes.get('forward_repeat')
    if not repeat or repeat['status'] != 'ok':
        return {'status': 'order_conflict_pending', 'relation': None, 'order_conflict': True}
    third = repeat['judgment']['normalized_relation']
    if third is not None:
        label, count = Counter(labels + [third]).most_common(1)[0]
        if count >= 2:
            return {'status': 'majority_after_conflict', 'relation': label, 'order_conflict': True}
    return {'status': 'unresolved_conflict', 'relation': None, 'order_conflict': True}


def verify_cumulative_ledger(path, prepared_preview):
    """Require the original ledger snapshot as an intact prefix, not a fresh budget."""
    snapshot = prepared_preview['budget_snapshot']
    if not snapshot.get('ledger_found') or not snapshot.get('ledger_sha256'):
        raise ValueError('Preparation had unknown prior spend; reprepare with the real original ledger')
    current = t.budget_snapshot(path)
    if not current['ledger_found']:
        raise ValueError('Original cumulative ledger missing; no API calls or budget reset allowed')
    import hashlib
    prefix = hashlib.sha256()
    found = False
    for line in Path(path).read_bytes().splitlines(keepends=True):
        prefix.update(line)
        if prefix.hexdigest() == snapshot['ledger_sha256']:
            found = True
    if not found:
        raise ValueError('Original cumulative ledger was replaced/truncated since preparation')
    return current


def load_run_inputs(args):
    directory, run_dir, output = (Path(p).resolve() for p in (args.input_dir, args.run_dir, args.output_dir))
    manifest, values = t.verify_bundle(directory)
    protocol = values['review_protocol.json']
    ledger_path = Path(protocol['budget_ledger_read_only_reference']).resolve()
    protected = (directory, ledger_path.parent, Path(manifest['trace_jsonl']).resolve().parent)
    for target in (run_dir, output):
        if any(target == p or target.is_relative_to(p) for p in protected):
            raise ValueError('New run/report must be outside prepared inputs, old paid audit and frozen trace pool')
    if run_dir == output or run_dir.is_relative_to(output) or output.is_relative_to(run_dir):
        raise ValueError('Run cache and report need separate non-nested directories')
    old_reports = ROOT / 'outputs/quality_assessment'
    if output.parent == old_reports and output.name.startswith('seed42_deepseek_flash_run01'):
        raise ValueError('Do not overwrite old quality audit reports')
    if args.max_pairs < 0 or args.max_pairs > len(values['blind_pairs.json']):
        raise ValueError('--max-pairs must be 0 (all) or at most the prepared pair count')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Timeout must be finite and positive')
    verify_cumulative_ledger(ledger_path, values['preparation_preview.json'])
    binding = {'protocol': 'training_edit_relation_judge_v1',
               'input_dir': str(directory), 'source_manifest_sha256': q.file_hash(directory / 'source_manifest.json'),
               'prepared_protocol_sha256': t.digest(protocol),
               'prepared_code_sha256': manifest['code_sha256'],
               'runner_code_sha256': q.file_hash(Path(__file__)),
               'shared_support_code_sha256': q.file_hash(Path(q.__file__)),
               'model': t.MODEL, 'base_url': 'https://api.deepseek.com',
               'max_tokens': protocol['max_tokens'], 'prompt_sha256': t.digest(t.SYSTEM_PROMPT),
               'shared_budget_ledger': str(ledger_path), 'cumulative_budget_limit_cny': '30',
               'max_attempts_per_order': 2, 'human_ground_truth': False,
               'selection_or_training_use': False, 'response_models_may_be_aliases': True,
               'price_checked_date': t.PRICE_DATE, 'input_cny_per_million': str(t.INPUT_PRICE),
               'output_cny_per_million': str(t.OUTPUT_PRICE)}
    marker = run_dir / 'judge_binding.json'
    if marker.exists():
        if q.read_json(marker) != binding:
            raise ValueError('Run binding/code/prompt changed; do not overwrite this run')
    elif run_dir.exists():
        raise FileExistsError('Refusing existing unbound run directory, even if empty')
    return directory, run_dir, output, values, ledger_path, binding


def read_vote(run_dir, binding, pair, order):
    path = run_dir / 'votes' / f"{pair['pair_id']}_{order}.json"
    vote = q.load_vote(path, q.fingerprint(binding), pair, order)
    if vote:
        body = t.build_request(pair, order, binding['max_tokens'])
        if vote.get('request_sha256') != t.digest(body):
            raise ValueError('Cached request differs from prepared prompt/text')
        if vote['status'] == 'ok' and parse_judgment(vote['content'], pair, order) != vote['judgment']:
            raise ValueError('Cached parsed judgment differs from response')
    return vote


def obtain_vote(pair, order, args, run_dir, binding, ledger, transport):
    cached = read_vote(run_dir, binding, pair, order)
    if cached and cached['status'] == 'ok':
        return cached
    body = t.build_request(pair, order, binding['max_tokens'])
    # Independent of cache directory: changing --run-dir must not reset attempts.
    request_id = t.digest({'binding': binding, 'pair_id': pair['pair_id'], 'order': order})
    vote = cached
    for number in range(2):
        attempt = f'training-edit:{request_id}:{number}'
        if any(e['kind'] == 'reserve' and e['attempt'] == attempt for e in ledger.events):
            continue
        reserve = t.reserve_bound(body)
        ledger.reserve(attempt, reserve)
        vote = {'pair_id': pair['pair_id'], 'order': order, 'status': 'error',
                'judge_binding_sha256': q.fingerprint(binding), 'request_sha256': t.digest(body),
                'attempt': attempt, 'time': q.utc_now()}
        fatal, stop = None, False
        try:
            response = transport(body, os.environ['DEEPSEEK_API_KEY'], args.timeout)
            # Account for reliable usage even if the completion is malformed.
            vote.update(response_id=response.get('id'), response_model=response.get('model'),
                        system_fingerprint=response.get('system_fingerprint'), usage=response.get('usage', {}))
            cost = q.usage_cost(vote['usage'])
            ledger.settle(attempt, cost)
            if cost > reserve or ledger.spent() > t.BUDGET_LIMIT:
                raise q.BudgetStop('Reserve exceeded; inspect official billing before further calls')
            choice = response['choices'][0]
            content = choice['message'].get('content') or ''
            vote.update(content=content, finish_reason=choice.get('finish_reason'))
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete completion is not a quality label')
            judgment = parse_judgment(content, pair, order)
            vote.update(status='ok', judgment=judgment)
        except q.BudgetStop:
            stop = True
            vote['error_type'] = 'BudgetStop'
        except HTTPError as exc:
            vote['error_type'] = f'HTTP_{exc.code}'
            if exc.code in (400, 401, 402, 403, 404):
                fatal = exc.code
        except Exception as exc:
            vote['error_type'] = type(exc).__name__
            # No arbitrary transport exception text (could contain credentials).
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
                vote['validation_error'] = str(exc)
        vote['record_sha256'] = q.fingerprint(vote)
        q.immutable_json(run_dir / 'attempts' / f'{t.digest(attempt)}.json', vote)
        q.save_json(run_dir / 'votes' / f"{pair['pair_id']}_{order}.json", vote)
        if fatal:
            raise RuntimeError(f'API HTTP {fatal}; stopped without printing response body or credentials')
        if stop:
            raise q.BudgetStop('Token reserve exceeded; no further requests')
        if vote['status'] == 'ok':
            return vote
    return vote


def write_summary(run_dir, output, values, binding):
    marker = output / 'training_edit_summary_binding.json'
    if output.exists() and not marker.is_file():
        raise FileExistsError('Refusing an existing unbound report directory')
    output_binding = {'judge_binding_sha256': q.fingerprint(binding), 'run_dir': str(run_dir),
                      'source_manifest_sha256': binding['source_manifest_sha256']}
    if marker.exists() and q.read_json(marker) != output_binding:
        raise ValueError('Report is already bound to a different run')
    decisions, models = [], set()
    for pair, source in zip(values['blind_pairs.json'], values['private_pair_records.json']):
        votes = {order: read_vote(run_dir, binding, pair, order) for order in ORDERS}
        final = final_decision(votes)
        for vote in votes.values():
            if vote and isinstance(vote.get('response_model'), str):
                models.add(vote['response_model'])
        decisions.append({'pair_id': pair['pair_id'], 'attack': source['attack'],
                          'parent_essay_id': source['parent_essay_id'], 'record_id': source['record_id'],
                          **final, 'recorded_B0_step_gain': source['recorded_step_gain'],
                          'vote_statuses': {o: v['status'] if v else 'absent_or_unknown_attempt'
                                            for o, v in votes.items()},
                          'vote_decision_statuses': {o: v['judgment']['decision_status']
                                                    if v and v['status'] == 'ok' else None
                                                    for o, v in votes.items()},
                          'training_supervision_approved': False})
    rows = []
    for attack in t.ATTACKS:
        group = [d for d in decisions if d['attack'] == attack]
        counts = Counter(d['relation'] for d in group)
        stable = [d for d in group if d['status'] == 'order_consistent']
        rows.append({'attack': attack, 'n_pairs': len(group),
                     'resolved': sum(counts[label] for label in LABELS),
                     **{label: counts[label] for label in LABELS},
                     'unknown': counts[None], 'order_conflicts': sum(d['order_conflict'] for d in group),
                     'stable_only_counts': {label: sum(d['relation'] == label for d in stable) for label in LABELS},
                     'mean_recorded_B0_step_gain_by_relation': {
                         label: sum(float(d['recorded_B0_step_gain']) for d in group if d['relation'] == label)
                         / counts[label] if counts[label] else None for label in LABELS}})
    summary = {'protocol': binding['protocol'], 'role': 'training_source_feasibility_only',
               'n_pairs': len(decisions), 'resolved': sum(d['relation'] is not None for d in decisions),
               'decision_status_counts': dict(Counter(d['status'] for d in decisions)),
               'response_models': sorted(models), 'model_alias_warning': 'Response aliases are not immutable weights.',
               'quality_relation_count': 1, 'human_ground_truth': False,
               'training_supervision_approved': False, 'new_texts_generated': 0,
               'selection_bias': 'Existing accepted B0-positive-gain training steps, not population prevalence.',
               'interpretation': 'Not an ASR evaluation, not evidence that training or scoring improved. '
                                 'A quality direction does not specify a numeric score change.',
               'rows': rows}
    with q.RunLock(output):
        q.immutable_json(marker, output_binding)
        q.save_json(output / 'training_edit_quality_results.json', summary)
        q.save_json(output / 'pair_decisions.json', decisions)
        lines = ['# Existing training edits: quality-relation feasibility audit', '',
                 'One relation for the observed edit in full context; no separate local/global labels.',
                 'LLM-assisted generic criteria, not human ground truth or an official rubric.',
                 'Selected accepted B0-positive steps only; not quality prevalence or an ASR evaluation.',
                 'Unknown/error/mixed judgments are NOT unchanged. No training supervision approved.', '',
                 '| Attack | Pairs | Resolved | Improved | Unchanged | Worsened | Unknown | Order conflicts |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            lines.append('| ' + ' | '.join(str(row[k]) for k in
                         ('attack', 'n_pairs', 'resolved', 'improved', 'unchanged', 'worsened',
                          'unknown', 'order_conflicts')) + ' |')
        lines += ['', 'R, H and Injection are peer families. Injection subtypes each carry half a family weight.',
                  'Order-conflict majorities are flagged; stable-only counts are available in JSON.',
                  f"Resolved: {summary['resolved']}/{summary['n_pairs']}. Pending/failed pairs remain unknown."]
        (output / 'training_edit_quality_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[SUMMARY] {output / "training_edit_quality_summary.md"}', flush=True)
    return summary


def run(args, transport=None):
    directory, run_dir, output, values, ledger_path, binding = load_run_inputs(args)
    selected = values['blind_pairs.json'][:args.max_pairs] if args.max_pairs else values['blind_pairs.json']
    if args.dry_run:
        result = t.cost_preview(selected, binding['max_tokens'], ledger_path)
        result.update(selected_for_this_run=len(selected), prepared_total_pairs=len(values['blind_pairs.json']),
                      run_dir=str(run_dir), output_dir=str(output), note_execution='No writes or API calls.')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if not args.summarize_only and (not args.allow_upload or not os.environ.get('DEEPSEEK_API_KEY', '').strip()):
        raise ValueError('Paid run needs --allow-upload and local DEEPSEEK_API_KEY; never paste the key')
    if args.summarize_only and not (run_dir / 'judge_binding.json').is_file():
        raise FileNotFoundError('No bound judge run exists; run the explicit judge first')
    # Validate report ownership BEFORE any uploads, not only when reporting.
    marker = output / 'training_edit_summary_binding.json'
    if output.exists() and not marker.is_file():
        raise FileExistsError('Refusing existing unbound report directory')
    expected_output_binding = {'judge_binding_sha256': q.fingerprint(binding), 'run_dir': str(run_dir),
                               'source_manifest_sha256': binding['source_manifest_sha256']}
    if marker.exists() and q.read_json(marker) != expected_output_binding:
        raise ValueError('Report already bound to another judge run')
    stopped = False
    with ExitStack() as stack:
        stack.enter_context(q.RunLock(ledger_path.parent))
        # The old shared budget lock serializes original audit, repairs and this run.
        _, _, _, checked_values, _, checked_binding = load_run_inputs(args)
        if checked_binding != binding:
            raise ValueError('Inputs changed while acquiring budget lock')
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            q.immutable_json(run_dir / 'judge_binding.json', binding)
        stack.enter_context(q.RunLock(run_dir))
        q.immutable_json(run_dir / 'judge_binding.json', binding)
        ledger = q.BudgetLedger(ledger_path, 30)
        try:
            if not args.summarize_only:
                for index, pair in enumerate(selected, 1):
                    votes = {}
                    for order in ORDERS:
                        if order == 'forward_repeat' and final_decision(votes)['status'] != 'order_conflict_pending':
                            break
                        votes[order] = obtain_vote(pair, order, args, run_dir, binding, ledger,
                                                   transport or q.call_deepseek)
                    decision = final_decision(votes)
                    print(f"[PAIR {index}/{len(selected)}] {pair['pair_id'][:12]} {decision['status']} "
                          f"relation={decision['relation']}; used/reserved={ledger.spent():.6f}/30 CNY", flush=True)
        except q.BudgetStop as exc:
            stopped = True
            print(f'[BUDGET STOP] {exc}', flush=True)
        finally:
            q.save_json(run_dir / 'budget_status.json', {
                'limit_cny': 30, 'used_or_reserved_cny': str(ledger.spent()),
                'remaining_cny': str(max(Decimal(0), t.BUDGET_LIMIT - ledger.spent())),
                'shared_ledger': str(ledger_path),
                'note': 'Conservative cumulative accounting, not an invoice/account cap. '
                        'Includes old audit, repairs and training-edit feasibility calls.'})
            write_summary(run_dir, output, checked_values, binding)
    return 2 if stopped else 0


def main(summarize_only=False):
    parser = argparse.ArgumentParser(description='Judge existing training edits; no generation, attacks or training.')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-pairs', type=int, default=0, help='0=all prepared pairs; resume SAME run after a pilot.')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-upload', action='store_true', help='Authorize sending prepared essays to DeepSeek.')
    args = parser.parse_args()
    args.summarize_only = summarize_only
    try:
        return run(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as exc:
        parser.exit(2, f'[STOP] {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
