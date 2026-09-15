import tempfile
import unittest
from pathlib import Path
from mlm_guided.evaluate_aes_seed42_missing_transfers import command_for, selection_paths, validate_result
from paer.rhi_experiment_utils import save_json, bind_directory
from whitebox.eval_hotflip_defended import build_parser


class MissingMLMTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(n_steps=30, beam_size=1, max_candidates_per_step=16,
                           n_sample_pos=8, top_k_per_pos=2, success_threshold=.1,
                           mlm_model_name='answerdotai/ModernBERT-large', mlm_dtype='bfloat16',
                           mlm_max_length=8192, similarity_model_name='sentence-transformers/all-MiniLM-L6-v2',
                           minimum_cosine_similarity=.9, max_token_edit_rate=.05)
        self.reference = dict(data='data file.csv', seed=42, batch_size=4, dtype='float32', device='cuda', max_length=1024)

    def test_actual_parser_and_mlm_edit_budget(self):
        command = command_for('checkpoint path', Path('output dir'), self.reference, self.params, 1154)
        args = build_parser().parse_args(command[2:])
        self.assertEqual(args.attack, 'mlm_guided')
        self.assertEqual(args.mlm_max_token_edit_rate, .05)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.n_essays, 1154)
        self.assertEqual(args.mlm_dtype, 'bfloat16')

    def test_exact_five_targets(self):
        jobs = selection_paths(Path('outputs'), Path('run02'))
        self.assertEqual(len(jobs), 5)
        self.assertEqual(jobs['D-Injection'][1], 'gstep1200')
        self.assertIn('run02', str(jobs['PAER-without-routing'][0]))

    def test_reject_wrong_result_count(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            save_json(out/'asr_summary.json', [dict(attack='mlm_guided', n_essays=2)])
            with self.assertRaises(ValueError):
                validate_result(out, 1154, self.params)

    def test_validate_matching_result(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            save_json(out/'asr_summary.json', [dict(attack='mlm_guided', n_essays=1154, asr=.5)])
            save_json(out/'run_manifest.json', dict(seed=42, attack_parameters=self.params))
            self.assertEqual(validate_result(out, 1154, self.params)['asr'], .5)

    def test_foreign_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'old.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                bind_directory(p, {'new':True})
            self.assertEqual((p/'old.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
