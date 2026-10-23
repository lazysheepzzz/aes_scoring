"""Forward-only SAME-text comparison on existing training-edit quality pairs.

No API calls, new text generation, attacks, training or human-quality claims.
Standard-library dry-run/tests; torch/transformers are loaded only for scoring.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer import quality_relation_audit as q
from paer import training_quality_pairs as t
from paer import training_quality_judge as j

MODELS = ('b0', 'mixed', 'paer')
PROTOCOL = 'fixed_training_quality_pair_response_v1'


def load_audited_pairs(input_dir, judge_dir):
    manifest, values = t.verify_bundle(input_dir)
    binding = q.read_json(judge_dir / 'judge_binding.json')
    expected = {'protocol': 'training_edit_relation_judge_v1',
                'source_manifest_sha256': q.file_hash(input_dir / 'source_manifest.json'),
                'prepared_protocol_sha256': t.digest(values['review_protocol.json']),
                'prepared_code_sha256': manifest['code_sha256'],
                'runner_code_sha256': q.file_hash(Path(j.__file__)),
                'shared_support_code_sha256': q.file_hash(Path(q.__file__)),
                'model': t.MODEL, 'max_tokens': values['review_protocol.json']['max_tokens'],
                'prompt_sha256': t.digest(t.SYSTEM_PROMPT)}
    if any(binding.get(k) != value for k, value in expected.items()):
        raise ValueError('Judge binding differs from the frozen preparation/code/prompt')
    rows, snapshot = [], {}
    for pair, source in zip(values['blind_pairs.json'], values['private_pair_records.json']):
        votes = {}
        for order in j.ORDERS:
            path = judge_dir / 'votes' / f"{pair['pair_id']}_{order}.json"
            snapshot[path.name] = t.file_hash(path) if path.is_file() else None
            votes[order] = j.read_vote(judge_dir, binding, pair, order)
        rows.append({**pair, 'source': source, 'decision': j.final_decision(votes)})
    provenance = {'prepared_manifest_sha256': t.file_hash(input_dir / 'source_manifest.json'),
                  'judge_binding_sha256': t.file_hash(judge_dir / 'judge_binding.json'),
                  'vote_files_sha256_including_absent': snapshot}
    return manifest, rows, provenance


def checkpoint_files(path):
    path = Path(path).resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Local checkpoint/config missing: {path}')
    weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin'))
    if not weights:
        raise FileNotFoundError(f'Local model weights missing: {path}')
    # Include tokenizer data, configs, weight shards and PAER heads, not only the folder name.
    files = sorted(set(weights) | set(path.glob('paer_*.pt')) | set(path.glob('*.json'))
                   | set(path.glob('*.model')) | set(path.glob('*.txt')) | set(path.glob('*.spm'))
                   | set(path.glob('*.vocab')) | set(path.glob('*.merges')))
    return {p.name: t.file_hash(p) for p in files}


def selected_checkpoint(selection_path):
    selection_path = Path(selection_path).resolve()
    value = q.read_json(selection_path)
    path = Path(value['checkpoint_path'])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not value.get('checkpoint_name') or value.get('max_checkpoint_step') != 1400:
        raise ValueError(f'Expected the frozen uniform-30 selection (max step 1400): {selection_path}')
    selection_binding_path = selection_path.parent / 'rhi_run_binding.json'
    frozen = q.read_json(selection_binding_path)
    if (set(frozen.get('selection_attacks', [])) != {'rudimentary', 'hotflip', 'injection_family'}
            or frozen.get('mlm_used_for_selection') is not False
            or any(frozen.get('arguments', {}).get(k) != 30 for k in
                   ('selection_steps', 'rudimentary_selection_steps', 'injection_selection_steps'))):
        raise ValueError('Recorded selection binding is not uniform-30 RHI with MLM excluded')
    weights = frozen.get('candidate_weights', {}).get(value['checkpoint_name'])
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Selected checkpoint has no recorded selection-time weight identity')
    # Record the selection, do not re-select a checkpoint on these diagnostic pairs.
    return path, {'path': str(selection_path), 'sha256': t.file_hash(selection_path),
                  'checkpoint_name': value['checkpoint_name'],
                  'selection_binding_sha256': t.file_hash(selection_binding_path),
                  'selected_checkpoint_files_sha256': weights}


def prepare_run(args):
    input_dir, judge_dir, output = (Path(p).resolve() for p in (args.input_dir, args.judge_dir, args.out))
    if args.batch_size <= 0 or not 16 <= args.max_length <= 1024:
        raise ValueError('Use positive batch-size and max-length 16..1024')
    if not math.isfinite(args.numeric_epsilon) or not 0 < args.numeric_epsilon <= .01:
        raise ValueError('numeric-epsilon must be finite, positive and <= 0.01 (numerical tolerance, not quality)')
    manifest, rows, provenance = load_audited_pairs(input_dir, judge_dir)
    ledger_parent = Path(t.read_json(input_dir / 'review_protocol.json')['budget_ledger_read_only_reference']).resolve().parent
    protected = (input_dir, judge_dir, ledger_parent, Path(manifest['trace_jsonl']).resolve().parent,
                 Path(manifest['train_csv']).resolve().parent, Path(manifest['valid_csv']).resolve().parent)
    mixed, mixed_selection = selected_checkpoint(args.mixed_selection)
    paer, paer_selection = selected_checkpoint(args.paer_selection)
    paths = {'b0': Path(args.b0_checkpoint).resolve(), 'mixed': mixed, 'paer': paer}
    for path in (*protected, *paths.values()):
        if output == path or output.is_relative_to(path):
            raise ValueError('Output must be outside frozen inputs, old audits, data and checkpoints')
    old_reports = ROOT / 'outputs/quality_assessment'
    if output.is_relative_to(old_reports):
        first = output.relative_to(old_reports).parts[0] if output != old_reports else ''
        if first.startswith(('seed42_deepseek_flash_run01', 'training_edit_relations_seed42_judge01')):
            raise ValueError('Do not write into old quality reports')
    if len(set(paths.values())) != 3:
        raise ValueError('B0, Mixed and PAER must be distinct checkpoints')
    paer_config = q.read_json(paer / 'paer_config.json')
    if paer_config.get('model_type') != 'paer_aes_v3' or not (paer / 'paer_heads.pt').is_file():
        raise ValueError('Expected the frozen PAER-v3 checkpoint with paer_heads.pt')
    files = {name: checkpoint_files(path) for name, path in paths.items()}
    for name, selection in (('mixed', mixed_selection), ('paer', paer_selection)):
        if any(files[name].get(key) != value for key, value in selection['selected_checkpoint_files_sha256'].items()):
            raise ValueError('Checkpoint differs from its frozen selection-time identity')
    generation_verified = False
    pool_manifest = Path(manifest['trace_jsonl']).with_suffix('.manifest.json')
    if pool_manifest.is_file():
        if t.file_hash(pool_manifest) != manifest['source_hashes']['trace_manifest']:
            raise ValueError('Frozen pool manifest changed since pair preparation')
        generation_weights = q.read_json(pool_manifest).get('checkpoint')
        if isinstance(generation_weights, dict) and generation_weights:
            if any(files['b0'].get(key) != value for key, value in generation_weights.items()):
                raise ValueError('B0 differs from the recorded RHI generation checkpoint identity')
            generation_verified = True
    code_files = [Path(__file__), Path(t.__file__), Path(j.__file__), Path(q.__file__),
                  ROOT / 'text_scoring_adv_training/evaluation/aes/scorer.py',
                  ROOT / 'paer/modeling_paer_v3.py', ROOT / 'paer/modeling_paer.py']
    binding = {'protocol': PROTOCOL, 'role': 'training_in_sample_mechanism_diagnostic',
               'input_dir': str(input_dir), 'judge_dir': str(judge_dir), 'inputs': provenance,
               'checkpoints': {name: {'path': str(paths[name]), 'files_sha256': files[name]} for name in MODELS},
               'selections': {'mixed': mixed_selection, 'paer': paer_selection},
               'code_sha256': {str(path.relative_to(ROOT)): t.file_hash(path) for path in code_files},
               'batch_size': args.batch_size, 'max_length': args.max_length, 'dtype': args.dtype,
               'device': args.device, 'numeric_epsilon': args.numeric_epsilon,
               'primary_cohort': 'order_consistent AND worsened; all pairs retained separately',
               'token_visibility': 'Report all primary pairs AND common tokenizer-visible primary subset',
               'network_calls': 0, 'new_texts_generated': 0, 'training_started': False,
               'B0_matches_generation_manifest_verified': generation_verified,
               'human_ground_truth': False, 'not_independent_test': True,
               'not_adaptive_attack_evaluation': True, 'edited_text_has_no_human_score': True}
    marker = output / 'fixed_pair_run_binding.json'
    if marker.is_file():
        if q.read_json(marker) != binding:
            raise ValueError('Inputs/checkpoints/settings changed; choose a NEW output directory')
    elif output.exists():
        raise FileExistsError('Refusing existing unbound output directory, even if empty')
    return rows, paths, output, binding


def metrics(deltas, identical_inputs, epsilon):
    if not deltas:
        return {'n_pairs': 0, 'mean_delta': None, 'median_delta': None,
                'mean_positive_delta': None, 'increase_rate_gt_numeric_epsilon': None,
                'decrease_rate_lt_minus_numeric_epsilon': None, 'near_zero_rate': None,
                'increase_rate_ge_0_05': None, 'increase_rate_ge_0_1': None,
                'identical_model_input_pairs': 0}
    n = len(deltas)
    return {'n_pairs': n, 'mean_delta': statistics.mean(deltas), 'median_delta': statistics.median(deltas),
            'mean_positive_delta': statistics.mean(max(0., v) for v in deltas),
            'increase_rate_gt_numeric_epsilon': sum(v > epsilon for v in deltas) / n,
            'decrease_rate_lt_minus_numeric_epsilon': sum(v < -epsilon for v in deltas) / n,
            'near_zero_rate': sum(abs(v) <= epsilon for v in deltas) / n,
            'increase_rate_ge_0_05': sum(v >= .05 for v in deltas) / n,
            'increase_rate_ge_0_1': sum(v >= .1 for v in deltas) / n,
            'identical_model_input_pairs': sum(identical_inputs)}


def summarize_rows(rows, scores, epsilon):
    primary = [r for r in rows if r['decision']['status'] == 'order_consistent'
               and r['decision']['relation'] == 'worsened']
    def visible(row):
        return all(scores[m][row['text_A']]['input_sha256'] != scores[m][row['text_B']]['input_sha256']
                   for m in MODELS)
    groups = {'stable_worsened_all': primary,
              'stable_worsened_common_model_visible': [r for r in primary if visible(r)],
              'stable_improved': [r for r in rows if r['decision']['status'] == 'order_consistent'
                                  and r['decision']['relation'] == 'improved'],
              'stable_unchanged': [r for r in rows if r['decision']['status'] == 'order_consistent'
                                   and r['decision']['relation'] == 'unchanged'],
              'other_or_unresolved': [r for r in rows if r['decision']['status'] != 'order_consistent'],
              'all_prepared_pairs': rows}
    summary = {}
    for cohort, subset in groups.items():
        results = {}
        for model in MODELS:
            def measure(items):
                return metrics([scores[model][r['text_B']]['score'] - scores[model][r['text_A']]['score']
                                for r in items],
                               [scores[model][r['text_A']]['input_sha256'] == scores[model][r['text_B']]['input_sha256']
                                for r in items], epsilon)
            attacks = {attack: measure([r for r in subset if r['source']['attack'] == attack]) for attack in t.ATTACKS}
            fields = ('mean_delta', 'mean_positive_delta', 'increase_rate_gt_numeric_epsilon',
                      'decrease_rate_lt_minus_numeric_epsilon', 'near_zero_rate',
                      'increase_rate_ge_0_05', 'increase_rate_ge_0_1')
            macro = {}
            for field in fields:
                vv = [attacks[a][field] for a in t.ATTACKS]
                macro[field] = (vv[0] + vv[1] + (vv[2] + vv[3]) / 2) / 3 if all(v is not None for v in vv) else None
            results[model] = {'pooled_descriptive': measure(subset), 'by_attack': attacks,
                              'rhi_equal_family_macro': macro}
        summary[cohort] = {'n_pairs': len(subset), 'models': results}
    comparisons = {}
    for cohort in ('stable_worsened_all', 'stable_worsened_common_model_visible'):
        items = groups[cohort]
        differences = []
        for r in items:
            dd = {m: scores[m][r['text_B']]['score'] - scores[m][r['text_A']]['score'] for m in MODELS}
            differences.append(dd['paer'] - dd['mixed'])
        comparisons[cohort] = {'n_pairs': len(items),
            'mean_paired_paer_minus_mixed_delta': statistics.mean(differences) if differences else None,
            'fraction_paer_lower_delta_than_mixed': sum(v < -epsilon for v in differences) / len(items) if items else None}
    details = []
    for r in rows:
        per_model = {}
        for model in MODELS:
            aa, bb = scores[model][r['text_A']], scores[model][r['text_B']]
            per_model[model] = {'score_A': aa['score'], 'score_B': bb['score'], 'delta': bb['score'] - aa['score'],
                                'identical_model_inputs': aa['input_sha256'] == bb['input_sha256']}
        details.append({'pair_id': r['pair_id'], 'attack': r['source']['attack'],
                        'parent_essay_id': r['source']['parent_essay_id'], 'decision': r['decision'],
                        'primary_stable_worsened': r in primary, 'common_model_visible': visible(r),
                        'recorded_B0_step_gain': r['source']['recorded_step_gain'], 'models': per_model})
    return {'role': 'training_in_sample_mechanism_diagnostic', 'n_prepared_pairs': len(rows),
            'n_stable_worsened': len(primary), 'numeric_epsilon': epsilon,
            'cohorts': summary, 'paired_comparisons': comparisons,
            'preservation_of_legitimate_gains': 'NOT_EVALUABLE_NO_STABLE_IMPROVED_PAIRS'
                if not groups['stable_improved'] else 'LLM_WEAK_LABEL_DESCRIPTIVE_ONLY_NOT_HUMAN_VALIDATION',
            'interpretation': 'Same saved texts for every model; not adaptive ASR, human quality truth, '
                              'independent generalization or numeric quality calibration. '
                              'B0-positive-gain selection favors positive B0 responses. No modified-text QWK computed.'}, details


def check_scores(payload, texts):
    if set(payload) != set(texts):
        raise ValueError('Scored text set differs from the bound fixed pairs')
    for value in payload.values():
        if (not isinstance(value['score'], (int, float)) or isinstance(value['score'], bool)
                or not math.isfinite(value['score']) or not isinstance(value['input_sha256'], str)
                or len(value['input_sha256']) != 64 or type(value['encoded_length']) is not int
                or value['encoded_length'] <= 0):
            raise ValueError('Nonfinite score or invalid tokenized-input fingerprint')


def score_model(model_name, path, texts, args):
    import torch
    from tqdm.auto import tqdm
    from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the experiment server or explicitly select --device cpu')
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    scorer = AESScorer(path, device=args.device, dtype=dtype)
    try:
        result = {}
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), args.batch_size), desc=f'{model_name}: fixed texts',
                              unit='batch', dynamic_ncols=True, disable=args.no_progress):
                batch = texts[start:start + args.batch_size]
                encoded = scorer.tokenizer(batch, return_tensors='pt', padding=True,
                                            truncation=True, max_length=args.max_length)
                output = scorer.model(input_ids=encoded['input_ids'].to(scorer.device),
                                      attention_mask=encoded['attention_mask'].to(scorer.device))
                predictions = output.logits.reshape(-1).float().cpu().tolist()
                if len(predictions) != len(batch):
                    raise ValueError('Expected one raw regression score per essay')
                for i, (text, prediction) in enumerate(zip(batch, predictions)):
                    ids = encoded['input_ids'][i][encoded['attention_mask'][i].bool()].tolist()
                    result[text] = {'score': float(prediction), 'input_sha256': t.digest(ids), 'encoded_length': len(ids)}
                del output, encoded
        return result
    finally:
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(args, backend=None):
    rows, paths, output, binding = prepare_run(args)
    counts = Counter((r['decision']['status'], r['decision']['relation']) for r in rows)
    preview = {'mode': 'FORWARD_ONLY_FIXED_TEXTS', 'n_prepared_pairs': len(rows),
               'n_stable_worsened': counts[('order_consistent', 'worsened')],
               'n_stable_improved': counts[('order_consistent', 'improved')],
               'checkpoints': {m: str(paths[m]) for m in MODELS}, 'output_dir': str(output),
               'batch_size': args.batch_size, 'max_length': args.max_length, 'dtype': args.dtype,
               'one_model_loaded_at_a_time': True, 'api_calls': 0, 'attacks': 0, 'training': False,
               'B0_matches_generation_manifest_verified': binding['B0_matches_generation_manifest_verified'],
               'note': 'Training-source weak-label diagnostic, NOT an independent test or modified-text QWK.'}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print('[DRY RUN] No GPU/model loads or files written.', flush=True)
        return preview
    # All checkpoints are local; disable downloads/telemetry before ML imports.
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    texts = list(dict.fromkeys(text for r in rows for text in (r['text_A'], r['text_B'])))
    all_scores = {}
    # Serialize with the judge: never observe partially updated votes, never touch its budget.
    with q.RunLock(Path(args.judge_dir).resolve()):
        _, _, checked = load_audited_pairs(Path(args.input_dir).resolve(), Path(args.judge_dir).resolve())
        if checked != binding['inputs']:
            raise ValueError('Judge inputs changed while acquiring lock')
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            q.immutable_json(output / 'fixed_pair_run_binding.json', binding)
        with q.RunLock(output):
            q.immutable_json(output / 'fixed_pair_run_binding.json', binding)
            for index, model in enumerate(MODELS, 1):
                print(f'[MODEL {index}/3] {model}: {paths[model]}', flush=True)
                cache = output / f'{model}_fixed_text_scores.json'
                if cache.is_file():
                    saved = q.read_json(cache)
                    if (saved['binding_sha256'] != t.digest(binding) or saved['model'] != model
                            or t.digest(saved['scores']) != saved['scores_sha256']):
                        raise ValueError('Score cache binding/content changed')
                    payload = {item['text_sha256']: item['value'] for item in saved['scores']}
                    scores = {text: payload[t.digest(text)] for text in texts}
                    print('[REUSE] Verified same-input score cache', flush=True)
                else:
                    scores = (backend or score_model)(model, paths[model], texts, args)
                    check_scores(scores, texts)
                    if checkpoint_files(paths[model]) != binding['checkpoints'][model]['files_sha256']:
                        raise ValueError('Checkpoint changed during scoring; results not accepted')
                    items = [{'text_sha256': t.digest(text), 'value': scores[text]} for text in texts]
                    q.immutable_json(cache, {'binding_sha256': t.digest(binding), 'model': model,
                                            'scores': items, 'scores_sha256': t.digest(items)})
                check_scores(scores, texts)
                if all_scores:
                    if any(scores[text]['input_sha256'] != all_scores['b0'][text]['input_sha256'] for text in texts):
                        raise ValueError('Models received different tokenized inputs; cannot claim matched input comparison')
                all_scores[model] = scores
            _, _, final_inputs = load_audited_pairs(Path(args.input_dir).resolve(), Path(args.judge_dir).resolve())
            if final_inputs != binding['inputs']:
                raise ValueError('Audit changed during scoring; choose a new run after freezing labels')
            summary, details = summarize_rows(rows, all_scores, args.numeric_epsilon)
            q.save_json(output / 'fixed_quality_pair_results.json', summary)
            q.save_json(output / 'fixed_quality_pair_details.json', details)
            lines = ['# Fixed-text quality-response diagnostic', '',
                     'TRAINING-source, in-sample diagnostic; LLM labels are not human ground truth.',
                     'Every model scores the SAME saved A/B texts. No attacks, API calls or training.',
                     'B0-positive-gain selection biases this sample. Results are not generalization evidence.',
                     'No modified-text QWK: the original human label is not copied to edited texts.',
                     f'Prepared pairs: {len(rows)}. Strict stable-worsened pairs: {summary["n_stable_worsened"]}.',
                     f'Numerical near-zero tolerance: {args.numeric_epsilon}; not a quality threshold.', '']
            for cohort in ('stable_worsened_all', 'stable_worsened_common_model_visible'):
                group = summary['cohorts'][cohort]
                lines += [f'## {cohort} (n={group["n_pairs"]})', '',
                          '| Model | Mean delta | Increase rate (>epsilon) | Increase rate (>=0.1) | Identical input pairs |',
                          '| --- | ---: | ---: | ---: | ---: |']
                def fmt(value):
                    return f'{value:.6f}' if value is not None else 'NA'
                for model in MODELS:
                    mm = group['models'][model]['pooled_descriptive']
                    lines.append(f'| {model} | {fmt(mm["mean_delta"])} | {fmt(mm["increase_rate_gt_numeric_epsilon"])} '
                                 f'| {fmt(mm["increase_rate_ge_0_1"])} | {mm["identical_model_input_pairs"]} |')
                lines += ['', 'Equal-family RHI macro = (R + H + (I-external + I-self-dup)/2)/3.', '',
                          '| Model | RHI macro mean delta | RHI macro increase rate (>epsilon) | RHI macro increase rate (>=0.1) |',
                          '| --- | ---: | ---: | ---: |']
                for model in MODELS:
                    mm = group['models'][model]['rhi_equal_family_macro']
                    lines.append(f'| {model} | {fmt(mm["mean_delta"])} | {fmt(mm["increase_rate_gt_numeric_epsilon"])} '
                                 f'| {fmt(mm["increase_rate_ge_0_1"])} |')
                if cohort == 'stable_worsened_all':
                    lines += ['', '| Model | Attack | n | Mean delta | Increase rate (>epsilon) |',
                              '| --- | --- | ---: | ---: | ---: |']
                    for model in MODELS:
                        for attack, mm in group['models'][model]['by_attack'].items():
                            lines.append(f'| {model} | {attack} | {mm["n_pairs"]} | {fmt(mm["mean_delta"])} '
                                         f'| {fmt(mm["increase_rate_gt_numeric_epsilon"])} |')
                lines += ['', 'Pooled values are descriptive; do not average four attack columns equally.', '']
            lines += ['Legitimate-gain preservation: ' + summary['preservation_of_legitimate_gains'],
                      'All 60 pairs, including conflict/unknown cases, remain in the separate detail/cohort outputs.']
            (output / 'fixed_quality_pair_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[SAVED] {output / "fixed_quality_pair_summary.md"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=j.DEFAULT_INPUT)
    parser.add_argument('--judge-dir', type=Path, default=j.DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/quality_assessment/fixed_training_quality_pair_response_seed42_run01')
    parser.add_argument('--b0-checkpoint', type=Path, default=ROOT / 'deberta_checkpoints/fold0_best')
    parser.add_argument('--mixed-selection', type=Path, default=ROOT / 'outputs/aes_mixed_at_rhi_checkpoint_selection_uniform30_seed42/best_checkpoint.json')
    parser.add_argument('--paer-selection', type=Path, default=ROOT / 'outputs/aes_paer_rhi_v3_checkpoint_selection_uniform30_seed42/best_checkpoint.json')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--numeric-epsilon', type=float, default=1e-5)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f'[STOP] {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
