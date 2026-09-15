"""Fill five seed42 MLM cells using frozen selections; never train or select."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import bind_directory, checkpoint_identity, read_json, save_json, sha256

PARAMETERS = {key: '--' + key.replace('_', '-') for key in (
    'n_steps', 'beam_size', 'max_candidates_per_step', 'n_sample_pos', 'top_k_per_pos',
    'success_threshold', 'mlm_model_name', 'mlm_dtype', 'mlm_max_length',
    'similarity_model_name', 'minimum_cosine_similarity')}
PARAMETERS['max_token_edit_rate'] = '--mlm-max-token-edit-rate'


def selection_paths(outputs, ablations):
    return {
        'D-HotFlip': (outputs / 'aes_hotflip_checkpoint_selection_seed42/best_checkpoint.json', 'gstep600'),
        'D-Rudimentary-v2': (outputs / 'aes_rudimentary_defense_v2_checkpoint_selection_seed42/best_checkpoint.json', 'gstep400'),
        'D-Injection': (outputs / 'aes_injection_checkpoint_selection_seed42/best_checkpoint.json', 'gstep1200'),
        'PAER-without-token-localization': (ablations / 'without_token_localization/selection/best_checkpoint.json', 'gstep400'),
        'PAER-without-routing': (ablations / 'without_routing/selection/best_checkpoint.json', 'gstep400'),
    }


def command_for(checkpoint, out, reference, params, n):
    command = [sys.executable, str(ROOT / 'whitebox/evaluate_aes_checkpoint.py'),
               '--checkpoint', str(checkpoint), '--out', str(out), '--attack', 'mlm_guided',
               '--current-python', '--valid', reference['data'], '--n-essays', str(n)]
    for key in ('seed', 'batch_size', 'dtype', 'device', 'max_length'):
        command += ['--' + key.replace('_', '-'), str(reference[key])]
    for key, flag in PARAMETERS.items():
        command += [flag, str(params[key])]
    return command


def validate_result(out, n, params):
    metrics = read_json(out / 'asr_summary.json')
    if len(metrics) != 1 or metrics[0]['attack'] != 'mlm_guided' or metrics[0]['n_essays'] != n:
        raise ValueError(f'Wrong attack or sample count: {out}')
    actual = read_json(out / 'run_manifest.json')
    if actual['seed'] != 42 or any(actual['attack_parameters'][k] != params[k] for k in PARAMETERS):
        raise ValueError(f'Actual MLM protocol differs: {out}')
    return metrics[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-evaluation', type=Path, default=ROOT / 'outputs/aes_rhi_evaluation_development_seed42')
    p.add_argument('--ablation-results', type=Path, default=ROOT / 'outputs/aes_paer_rhi_v3_retrained_ablations_seed42_run02')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/aes_seed42_missing_mlm_transfers')
    p.add_argument('--n-essays', type=int, help='Smoke limit only; default retains the full reference sample count')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    reference = read_json(args.reference_evaluation / 'rhi_run_binding.json')
    manifest = read_json(args.reference_evaluation / 'paer/mlm_guided/run_manifest.json')
    params = manifest['attack_parameters']
    if reference['seed'] != 42 or manifest['seed'] != 42 or manifest['attack'] != 'mlm_guided':
        raise ValueError('Expected the completed seed42 MLM reference')
    if manifest['n_essays'] != reference['n_essays'] or sha256(Path(manifest['data'])) != reference['data_sha256']:
        raise ValueError('Reference MLM data/count differs from binding')
    if sha256(Path(reference['data'])) != reference['data_sha256']:
        raise ValueError('Reference data changed')
    n = args.n_essays if args.n_essays is not None else reference['n_essays']
    if not 0 < n <= reference['n_essays']:
        raise ValueError('Invalid n-essays')
    jobs = {}
    for model, (path, expected) in selection_paths(ROOT / 'outputs', args.ablation_results).items():
        selected = read_json(path)
        checkpoint = Path(selected['checkpoint_path'])
        if selected['checkpoint_name'] != expected or checkpoint.name != expected:
            raise ValueError(f'{model}: selection differs from the reported seed42 table; inspect before running')
        identity = checkpoint_identity(checkpoint)
        selection_binding = path.parent / 'rhi_run_binding.json'
        if selection_binding.exists():
            if identity != read_json(selection_binding)['candidate_weights'][expected]:
                raise ValueError(f'Checkpoint changed since selection: {model}')
        jobs[model] = {'checkpoint': str(checkpoint), 'files': identity,
                       'selection_sha256': sha256(path), 'selected_clean_qwk': selected['clean_qwk']}
    code_files = list((ROOT / 'text_scoring_adv_training/evaluation/aes').rglob('*.py')) + list((ROOT / 'paer').glob('modeling_paer*.py')) + [Path(__file__), ROOT / 'whitebox/evaluate_aes_checkpoint.py', ROOT / 'whitebox/eval_hotflip_defended.py']
    protocol = {'reference': reference, 'mlm_parameters': params, 'jobs': jobs,
                'n_essays': n, 'smoke': n < reference['n_essays'], 'mlm_used_for_selection': False,
                'code_hashes': {str(f.relative_to(ROOT)): sha256(f) for f in code_files}}
    if not args.dry_run:
        bind_directory(args.output_dir, protocol)
    rows = []
    for i, (model, job) in enumerate(jobs.items(), 1):
        out = args.output_dir / model
        command = command_for(job['checkpoint'], out, reference, params, n)
        print(f'[{i}/5] {model}\n{ subprocess.list2cmdline(command)}', flush=True)
        if args.dry_run:
            continue
        marker = out / 'completed_result_hashes.json'
        if marker.exists():
            if any(sha256(out / name) != digest for name, digest in read_json(marker).items()):
                raise ValueError(f'Completed result modified: {out}')
        else:
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f'Partial output preserved: {out}; choose a new --output-dir')
            # One child at a time: release DeBERTa/MLM GPU allocations between jobs.
            subprocess.run(command, check=True)
            validate_result(out, n, params)
            required = ('clean_qwk.json', 'asr_summary.json', 'run_manifest.json', 'mlm_guided_details.json')
            save_json(marker, {name: sha256(out / name) for name in required})
        metrics = validate_result(out, n, params)
        clean = read_json(out / 'clean_qwk.json')
        rows.append({'model': model, 'checkpoint': job['checkpoint'], 'clean': clean, 'metrics': metrics})
        save_json(args.output_dir / 'mlm_transfer_results.json', {'protocol': protocol, 'results': rows})
        lines = ['# Seed42 missing MLM transfer evaluations', '',
                 f"Role: {reference['evaluation_role']}; n={n}; smoke={protocol['smoke']}; completed={len(rows)}/5", '',
                 '| Model | Clean QWK | MLM ASR | Mean delta |', '| --- | --- | --- | --- |']
        lines += [f"| {r['model']} | {r['clean']['qwk']:.4f} | {r['metrics']['asr']:.4f} | {r['metrics']['avg_delta']:.4f} |" for r in rows]
        (args.output_dir / 'mlm_transfer_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
