"""Rejudge ONLY unresolved evidence-quotation failures; preserve the original audit.

Shares the parent's 30-CNY ledger. Adds an explicit short-quotation instruction,
not a different quality criterion. Original votes and original summaries remain intact.
"""
from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paer import quality_relation_audit as q

QUOTATION_REMINDER = (
    '\nQuotation formatting reminder: copy short, CONTIGUOUS evidence spans '
    '(preferably 3-8 words) exactly from the supplied A and B strings. '
    'Preserve original spelling, punctuation, capitalization and whitespace. '
    'Never join separated passages or silently correct student errors in quotes. '
    'For this rejudgment provide a nonempty exact span from EACH version; '
    'never fabricate one. Independently reassess overall '
    'quality using the SAME three criteria and labels above; do not infer a '
    'desired label or continue retrying until any preferred label appears.'
)


def load_parent(directory):
    manifest, records, pairs = q.verify_inputs(directory)
    binding = q.read_json(directory / 'judge_binding.json')
    if binding['code_sha256'] != q.file_hash(Path(q.__file__)):
        raise ValueError('Parent judge code changed; do not rejudge using a silently changed protocol')
    if not (directory / 'budget_ledger.jsonl').is_file():
        raise ValueError('Original budget ledger is required; refusing a budget reset')
    binding_hash = q.fingerprint(binding)
    originals, vote_hashes, targets = {}, {}, []
    for pair in pairs:
        votes = {}
        for order in ('forward', 'reverse', 'forward_repeat'):
            name = f"{pair['pair_id']}_{order}.json"
            path = directory / 'votes' / name
            votes[order] = q.load_vote(path, binding_hash, pair, order)
            if path.exists():
                vote_hashes[name] = q.file_hash(path)
        originals[pair['pair_id']] = votes
        if q.final_relation(pair, votes).get('relation') is not None:
            continue
        for order, vote in votes.items():
            if not vote or vote['status'] == 'ok':
                continue
            try:
                q.usage_cost(vote.get('usage', {}))
                q.parse_judgment(vote.get('content', ''), pair, order)
            except (ValueError, KeyError, TypeError) as exc:
                if str(exc) == 'Evidence not present in supplied essay':
                    targets.append({'pair_id': pair['pair_id'], 'order': order,
                                    'original_vote_sha256': vote_hashes[f"{pair['pair_id']}_{order}.json"]})
    repair_binding = {
        'protocol': 'quotation_failure_rejudging_v1', 'parent_audit': str(directory),
        'parent_source_manifest_sha256': q.file_hash(directory / 'source_manifest.json'),
        'parent_judge_binding_sha256': binding_hash, 'original_vote_file_hashes': vote_hashes,
        'initial_failed_orders': targets, 'quotation_instruction': QUOTATION_REMINDER,
        'helper_code_sha256': q.file_hash(Path(__file__)),
        'max_attempts_per_added_order': 2, 'shared_budget_cny': 30,
        'max_tokens': binding['max_tokens'], 'quality_criteria_changed': False,
        'third_vote_rule': 'Only if repaired primary votes disagree and original tiebreak is absent/invalid',
    }
    return manifest, records, pairs, binding, originals, repair_binding


def extra_body(pair, order, max_tokens):
    body = q.request_body(pair, order, max_tokens)
    body['messages'][0]['content'] += QUOTATION_REMINDER
    return body


def get_repair_vote(pair, order, args, repair_binding, parent_binding, ledger, transport):
    binding_hash = q.fingerprint(parent_binding)
    repair_hash = q.fingerprint(repair_binding)
    path = args.repair_dir / 'votes' / f"{pair['pair_id']}_{order}.json"
    vote = q.load_vote(path, binding_hash, pair, order)
    if vote and vote.get('repair_binding_sha256') != repair_hash:
        raise ValueError('Repair cache protocol changed')
    if vote and vote['status'] == 'ok':
        return vote
    body = extra_body(pair, order, parent_binding['max_tokens'])
    request_id = q.fingerprint({'repair': repair_hash, 'pair_id': pair['pair_id'], 'order': order})
    for number in range(2):
        attempt = f'repair:{request_id}:{number}'
        if any(e['kind'] == 'reserve' and e['attempt'] == attempt for e in ledger.events):
            continue
        reserve = q.reserve_cost(body)
        ledger.reserve(attempt, reserve)
        vote = {'pair_id': pair['pair_id'], 'order': order, 'status': 'error',
                'judge_binding_sha256': binding_hash, 'repair_binding_sha256': repair_hash,
                'request_sha256': q.fingerprint(body), 'attempt': attempt, 'time': q.utc_now(),
                'note': 'Parent audit binding plus explicit quotation-only reminder; see repair_binding.json'}
        fatal, overrun = None, False
        try:
            response = transport(body, os.environ['DEEPSEEK_API_KEY'], args.timeout)
            choice = response['choices'][0]
            text = choice['message'].get('content') or ''
            vote.update(response_id=response.get('id'), response_model=response.get('model'),
                        system_fingerprint=response.get('system_fingerprint'), usage=response.get('usage', {}),
                        content=text, finish_reason=choice.get('finish_reason'))
            cost = q.usage_cost(vote['usage'])
            ledger.settle(attempt, cost)
            if cost > reserve or ledger.spent() > 30:
                overrun = True
                raise q.BudgetStop('Reserve exceeded; inspect official billing before any further calls')
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete completion is not a quality label')
            judgment = q.parse_judgment(text, pair, order)
            if not judgment['evidence_A'].strip() or not judgment['evidence_B'].strip():
                raise ValueError('Rejudgment requires nonempty exact evidence from both versions')
            vote.update(status='ok', judgment=judgment)
        except q.BudgetStop:
            overrun = True
            vote['error_type'] = 'BudgetStop'
        except HTTPError as exc:
            vote['error_type'] = f'HTTP_{exc.code}'
            if exc.code in (400, 401, 402, 403, 404):
                fatal = exc.code
        except Exception as exc:
            vote['error_type'] = type(exc).__name__
            if isinstance(exc, ValueError):
                vote['validation_error'] = str(exc)
        vote['record_sha256'] = q.fingerprint(vote)
        q.save_json(args.repair_dir / 'attempts' / f'{q.fingerprint(attempt)}.json', vote)
        q.save_json(path, vote)
        if fatal:
            raise RuntimeError(f'API HTTP {fatal}; stopped without logging response body or credentials')
        if overrun:
            raise q.BudgetStop('Token reserve exceeded')
        if vote['status'] == 'ok':
            return vote
    return vote


def write_derived_summary(args, pairs, originals, parent_binding, repair_binding):
    """Build a clearly marked derived view; never patch original response files."""
    view = args.repair_dir / 'derived_readonly_view'
    marker = view / 'DERIVED_VIEW_NOT_FOR_API.json'
    if view.exists() and any(view.iterdir()) and not marker.exists():
        raise FileExistsError('Refusing nonempty unbound derived-view directory')
    q.immutable_json(marker, {'repair_binding_sha256': q.fingerprint(repair_binding),
                             'not_an_api_run': True, 'all_charges_in_parent_ledger': True})
    for name in ('source_manifest.json', 'private_pair_records.json', 'blind_pairs.json', 'judge_binding.json'):
        q.immutable_json(view / name, q.read_json(args.audit_dir / name))
    replacements = []
    for pair in pairs:
        for order in ('forward', 'reverse', 'forward_repeat'):
            name = f"{pair['pair_id']}_{order}.json"
            replacement = q.load_vote(args.repair_dir / 'votes' / name,
                                      q.fingerprint(parent_binding), pair, order)
            if replacement and replacement.get('repair_binding_sha256') != q.fingerprint(repair_binding):
                raise ValueError('Repair vote binding changed')
            original = originals[pair['pair_id']][order]
            chosen = replacement if replacement and replacement['status'] == 'ok' else original
            if chosen:
                q.save_json(view / 'votes' / name, chosen)
            if replacement and replacement['status'] == 'ok':
                replacements.append({'pair_id': pair['pair_id'], 'order': order,
                                     'repair_record_sha256': replacement['record_sha256'],
                                     'original_status': original['status'] if original else 'absent'})
    q.summarize(argparse.Namespace(audit_dir=view, output_dir=args.output_dir))
    q.save_json(args.output_dir / 'evidence_repair_provenance.json', {
        'parent_audit': str(args.audit_dir), 'repair_binding_sha256': q.fingerprint(repair_binding),
        'original_votes_preserved': True, 'quality_labels_not_manually_rewritten': True,
        'quote_check_not_relaxed': True, 'rejudged_orders': replacements,
        'all_costs_in_parent_budget_ledger': True})
    with (args.output_dir / 'quality_relation_summary.md').open('a', encoding='utf-8') as stream:
        stream.write('\nThis is a derived report including targeted quotation-failure rejudgments. '
                     'Original responses/report remain intact; see evidence_repair_provenance.json.\n')


def run(args, transport=None):
    args.audit_dir = args.audit_dir.resolve()
    args.repair_dir = (args.repair_dir or args.audit_dir / 'evidence_repair_run01').resolve()
    args.output_dir = args.output_dir.resolve()
    if args.timeout <= 0:
        raise ValueError('Timeout must be positive')
    if args.repair_dir == args.audit_dir or args.repair_dir.is_relative_to(args.audit_dir / 'votes'):
        raise ValueError('Repair artifacts must not overwrite original audit/votes')
    if args.output_dir == args.audit_dir or args.output_dir.is_relative_to(args.audit_dir):
        raise ValueError('Put derived report under a NEW outputs directory')
    if args.output_dir == q.ROOT / 'outputs/quality_assessment/seed42_deepseek_flash_run01':
        raise ValueError('Do not overwrite the original summary directory')
    _, _, pairs, parent, originals, binding = load_parent(args.audit_dir)
    target_ids = {target['pair_id'] for target in binding['initial_failed_orders']}
    if args.dry_run:
        ledger = q.BudgetLedger(args.audit_dir / 'budget_ledger.jsonl', 30)
        print(q.packed({'initial_failed_orders': binding['initial_failed_orders'],
                        'affected_unique_pairs': len(target_ids), 'shared_used_or_reserved_cny': str(ledger.spent()),
                        'max_additional_tiebreaks': len(target_ids),
                        'network_calls': 0, 'original_votes_modified': False}))
        return
    if not args.summarize_only and (not args.allow_upload or not os.environ.get('DEEPSEEK_API_KEY')):
        raise ValueError('Need --allow-upload and local DEEPSEEK_API_KEY for paid rejudging')
    if args.repair_dir.exists() and any(args.repair_dir.iterdir()) and not (args.repair_dir / 'repair_binding.json').exists():
        raise FileExistsError('Refusing nonempty unbound repair directory')
    with q.RunLock(args.audit_dir):
        # Recheck parent snapshots AFTER obtaining its budget lock.
        _, _, pairs, parent, originals, checked = load_parent(args.audit_dir)
        if checked != binding:
            raise ValueError('Parent changed while acquiring lock')
        q.immutable_json(args.repair_dir / 'repair_binding.json', binding)
        ledger = q.BudgetLedger(args.audit_dir / 'budget_ledger.jsonl', 30)
        try:
            if not args.summarize_only:
                for pair in pairs:
                    if pair['pair_id'] not in target_ids:
                        continue
                    votes = dict(originals[pair['pair_id']])
                    for order in ('forward', 'reverse', 'forward_repeat'):
                        if votes.get(order) and votes[order]['status'] == 'ok':
                            continue
                        eligible = any(t['pair_id'] == pair['pair_id'] and t['order'] == order
                                       for t in binding['initial_failed_orders'])
                        tiebreak = order == 'forward_repeat' and q.final_relation(pair, votes)['status'] == 'order_conflict_pending'
                        if eligible or tiebreak:
                            repaired = get_repair_vote(pair, order, args, binding, parent, ledger,
                                                       transport or q.call_deepseek)
                            if repaired:
                                votes[order] = repaired
                    print(f"[REPAIR] {pair['pair_id'][:12]} {q.final_relation(pair, votes)['status']}; "
                          f'used/reserved={ledger.spent():.4f}/30 CNY', flush=True)
        except q.BudgetStop as exc:
            print(f'[BUDGET STOP] {exc}', flush=True)
        finally:
            q.save_json(args.audit_dir / 'budget_status.json', {
                'limit_cny': 30, 'used_or_reserved_cny': str(ledger.spent()),
                'remaining_cny': str(max(Decimal(0), Decimal(30) - ledger.spent())),
                'note': 'Conservative estimate, not official bill. Includes original audit AND evidence rejudging.'})
            write_derived_summary(args, pairs, originals, parent, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=q.ROOT / 'artifacts/quality_assessment/seed42_deepseek_flash_run01')
    parser.add_argument('--repair-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=q.ROOT / 'outputs/quality_assessment/seed42_deepseek_flash_run01_evidence_repair01')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-upload', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
