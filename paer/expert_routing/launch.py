"""CPU-only input validation and opt-in expert-training launchers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from paer.aes_rh_training_launcher import build_parser as old_parser, build_config, MIXED_AT_RH, PAER_RH_V3
from paer.rhi_experiment_utils import ROOT, RHI_ATTACKS, checkpoint_identity, read_json, sha256, validate_trace_groups

EXPERT_ONLY = 'low_rank_expert_rhi'
EXPERT_PAER = 'paer_expert_rhi'


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        text_key = next((k for k in ('full_text', 'text') if k in fields), None)
        if text_key is None or 'score' not in fields:
            raise ValueError('Expected training text and score columns')
        rows = [{'text': r[text_key], 'score': float(r['score']),
                 'id': r.get('essay_id', r.get('id'))} for r in reader]
    if not rows or any(not r['text'].strip() or not math.isfinite(r['score']) for r in rows):
        raise ValueError('Empty or invalid CSV rows')
    return rows


def smoke_indices(rows, groups, limit, seed):
    """Select EXISTING rows, guaranteeing R:H:I=1:1:1 among attacked parents."""
    limit = min(limit, len(rows))
    if limit < 6:
        raise ValueError('Smoke needs at least six rows to cover all peer families')
    quota = max(6, int(limit * len(groups) / len(rows) / 6) * 6)
    quota = min(quota, limit // 6 * 6)
    quotas = dict(zip(RHI_ATTACKS, (quota // 3, quota // 3, quota // 6, quota // 6)))
    def order(key, index):
        return hashlib.sha256(f'{seed}|{key}|{index}'.encode()).hexdigest()
    selected = []
    for attack, count in quotas.items():
        candidates = [i for i, records in groups.items() if records[0]['attack'] == attack]
        candidates.sort(key=lambda i: order(attack, i))
        if len(candidates) < count:
            raise ValueError(f'Existing pool too small for balanced smoke {attack}; no new text generated')
        selected.extend(candidates[:count])
    clean = sorted((i for i in range(len(rows)) if i not in groups), key=lambda i: order('clean', i))
    if len(clean) < limit - len(selected):
        raise ValueError('Not enough clean-only rows for the requested balanced smoke; use a smaller limit')
    return sorted(selected + clean[:limit - len(selected)])


def parser_for(experiment):
    if experiment not in (EXPERT_ONLY, EXPERT_PAER):
        raise ValueError('Unknown expert experiment')
    backend = MIXED_AT_RH if experiment == EXPERT_ONLY else PAER_RH_V3
    parser = old_parser(backend)
    parser.description = f'Opt-in {experiment}; RoME-inspired, same existing RHI pool, no judge or new texts.'
    parser.set_defaults(output_dir=None,
        trace_jsonl=ROOT / 'artifacts/paer/rhi_training_pool_seed42/rhi_counterfactual_training_traces.jsonl')
    parser.add_argument('--expert-rank', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=3)
    parser.add_argument('--expert-layers', type=int, default=2)
    parser.add_argument('--expert-learning-rate', type=float, default=1e-4)
    parser.add_argument('--expert-route-weight', type=float, default=.1)
    parser.add_argument('--prototype-momentum', type=float, default=.9)
    parser.add_argument('--prototype-temperature', type=float, default=.1)
    parser.add_argument('--routing-scope', choices=('edited', 'all'), default='edited')
    parser.add_argument('--smoke', action='store_true', help='One epoch; balanced existing-row subset (160/32 defaults).')
    return parser, backend


def prepare_config(args, experiment, backend):
    if args.max_trace_records is not None:
        raise ValueError('Prefix trace truncation is forbidden; use --smoke')
    if not args.smoke and (args.max_train_samples is not None or args.max_valid_samples is not None):
        raise ValueError('Partial datasets require explicit --smoke')
    if args.smoke:
        args.num_epochs = 1
        args.max_train_samples = 160 if args.max_train_samples is None else args.max_train_samples
        args.max_valid_samples = 32 if args.max_valid_samples is None else args.max_valid_samples
    for name in ('num_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                 'gradient_accumulation_steps', 'eval_every', 'save_every',
                 'max_train_samples', 'max_valid_samples', 'routing_top_k'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('learning_rate', 'clean_loss_weight', 'adversarial_loss_weight',
                 'relative_loss_power', 'attribution_gain_scale', 'paer_head_learning_rate',
                 'correction_loss_beta', 'max_correction_target', 'max_token_evidence',
                 'adam_epsilon', 'localization_positive_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('weight_decay', 'inflation_tolerance', 'localization_loss_weight',
                 'clean_false_suppression_weight', 'correction_scale', 'correction_calibration_weight',
                 'clean_correction_weight', 'route_lift_loss_weight', 'attention_alignment_loss_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if (not math.isfinite(args.routing_risk_bias_init) or not 0 <= args.warmup_ratio <= 1
            or not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1
            or not 0 <= args.correction_scale <= 1):
        raise ValueError('Invalid optimizer/risk initialization settings')
    if args.output_dir is None:
        args.output_dir = ROOT / 'outputs' / f'{"smoke_" if args.smoke else ""}aes_{experiment}_seed{args.seed}'
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f'Use a NEW output directory, not an existing one: {args.output_dir}')
    if (args.expert_rank <= 0 or args.expert_layers <= 0 or not 2 <= args.num_experts <= 8
            or not 16 <= args.max_length <= 1024 or not 0 <= args.prototype_momentum < 1
            or not all(math.isfinite(v) and v > 0 for v in
                       (args.prototype_temperature, args.expert_learning_rate))
            or not math.isfinite(args.expert_route_weight) or args.expert_route_weight < 0):
        raise ValueError('Invalid expert configuration; max-length is limited to 1024 for this 24GB prototype')
    for key in ('checkpoint_path', 'train_csv', 'valid_csv', 'trace_jsonl'):
        setattr(args, key, getattr(args, key).resolve())
    protected = [args.checkpoint_path, args.train_csv.parent, args.valid_csv.parent, args.trace_jsonl.parent]
    if any(args.output_dir == p or args.output_dir.is_relative_to(p) for p in protected):
        raise ValueError('Output must not be inside data, checkpoints or the frozen pool')
    if (args.checkpoint_path / 'paer_config.json').exists():
        raise ValueError('Initialize both new experiments from B0, not PAER or another defense')
    base_config = read_json(args.checkpoint_path / 'config.json')
    if (base_config.get('model_type') != 'deberta-v2'
            or args.expert_layers > base_config.get('num_hidden_layers', 0)):
        raise ValueError('Expert v1 requires a compatible DeBERTa encoder with enough layers')
    config = build_config(args, backend)
    rows, valid = read_rows(args.train_csv), read_rows(args.valid_csv)
    if {' '.join(r['text'].split()) for r in rows} & {' '.join(r['text'].split()) for r in valid}:
        raise ValueError('Train/validation text overlap')
    train_ids, valid_ids = ({r['id'] for r in rr if r['id'] is not None} for rr in (rows, valid))
    if train_ids & valid_ids:
        raise ValueError('Train/validation ID overlap')
    manifest_path = args.trace_jsonl.with_suffix('.manifest.json')
    manifest = read_json(manifest_path)
    identity = checkpoint_identity(args.checkpoint_path)
    if (manifest.get('mlm_excluded') is not True or manifest['source_hashes']['train'] != sha256(args.train_csv)
            or manifest['trace_sha256'] != sha256(args.trace_jsonl) or manifest['checkpoint'] != identity
            or manifest['arguments']['max_length'] != args.max_length):
        raise ValueError('Frozen RHI pool training source, B0, trace hash or max-length mismatch')
    with args.trace_jsonl.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    groups, full_counts = validate_trace_groups(records, rows)
    indices = smoke_indices(rows, groups, args.max_train_samples, args.seed) if args.smoke else None
    selected = set(indices) if indices is not None else set(range(len(rows)))
    counts = dict(Counter(group[0]['attack'] for i, group in groups.items() if i in selected))
    config.update(experiment_name=experiment, training_attacks=list(RHI_ATTACKS),
        expert_use_paer=experiment == EXPERT_PAER, expert_rank=args.expert_rank,
        num_experts=args.num_experts, expert_layers=args.expert_layers,
        expert_learning_rate=args.expert_learning_rate, expert_route_weight=args.expert_route_weight,
        prototype_momentum=args.prototype_momentum, prototype_temperature=args.prototype_temperature,
        routing_scope=args.routing_scope, selected_train_row_indices=indices, expert_smoke=args.smoke)
    inputs = {'experiment_name': experiment, 'backend_training_mode': backend,
        'train_sha256': sha256(args.train_csv), 'valid_sha256': sha256(args.valid_csv),
        'trace_sha256': sha256(args.trace_jsonl), 'pool_manifest_sha256': sha256(manifest_path),
        'checkpoint': identity, 'essay_counts_by_attack': counts,
        'full_pool_essay_counts_by_attack': full_counts,
        'n_clean_rows': len(selected), 'n_attacked_rows': sum(counts.values()),
        'selected_train_row_indices': indices, 'mlm_used_for_training': False,
        'quality_judge_used': False, 'new_texts_generated': 0,
        'evaluation_status': 'smoke_engineering_only' if args.smoke else 'development; not an independent test',
        'routing': 'per-essay-normalized edited-token gates; detached EMA family prototypes across microbatches',
        'source_sha256': {name: sha256(ROOT / name) for name in (
            'paer/expert_routing/model.py', 'paer/expert_routing/prototypes.py',
            'paer/expert_routing/launch.py', 'paer/expert_routing/trainer.py',
            'paer/aes_rh_trainer.py', 'paer/modeling_paer_v3.py')},
        'not_full_RoME_reproduction': True}
    return config, inputs


def main(experiment):
    parser, backend = parser_for(experiment)
    args = parser.parse_args()
    try:
        config, inputs = prepare_config(args, experiment, backend)
        print(json.dumps({'config': config, 'inputs': inputs}, indent=2, ensure_ascii=False), flush=True)
        if args.dry_run:
            print('[DRY RUN] CPU verification only; no files, model loads, attacks, API calls or training.', flush=True)
            return 0
        # Never adopt even an empty pre-existing directory or overwrite a prior run.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in (('launcher_config.json', config), ('rhi_training_inputs.json', inputs)):
            with (args.output_dir / name).open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        env = os.environ.copy()
        if not args.online:
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        command = [sys.executable, str(ROOT / 'paer/expert_routing/trainer.py'), '--config',
                   str(args.output_dir / 'launcher_config.json')]
        print('[RUN] ' + subprocess.list2cmdline(command), flush=True)
        return subprocess.run(command, env=env, check=False).returncode
    except (ValueError, KeyError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f'[STOP] {exc}\n')
