from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_results.build_aes_main_experiment_matrix import collect_rows


class AESMainExperimentMatrixTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_collects_three_peer_families_and_separate_mlm(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            self._write(
                outputs / "eval_b0_seed42" / "clean_qwk.json",
                {"qwk": 0.8, "mae": 0.4},
            )
            self._write(
                outputs / "eval_b0_seed42" / "asr_summary.json",
                [{"attack": "hotflip", "asr": 0.9, "avg_delta": 0.2}],
            )
            self._write(
                outputs / "eval_rudimentary_b0_seed42" / "asr_summary.json",
                [{"attack": "rudimentary", "asr": 0.6, "avg_delta": 0.1}],
            )
            self._write(
                outputs
                / "eval_injection_b0_seed42"
                / "injection_family_summary.json",
                {"asr": 0.3, "avg_delta": 0.05},
            )
            self._write(
                outputs / "eval_mlm_b0_seed42" / "asr_summary.json",
                [{"attack": "mlm_guided", "asr": 0.5, "avg_delta": 0.08}],
            )

            row = collect_rows(outputs)[0]

            self.assertEqual(row["model"], "B0")
            self.assertAlmostEqual(row["rhi_macro_asr"], 0.6)
            self.assertEqual(row["mlm_guided_asr"], 0.5)

    def test_missing_experiments_remain_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = collect_rows(Path(temporary))[0]
            self.assertIsNone(row["clean_qwk"])
            self.assertIsNone(row["rhi_macro_asr"])
            self.assertIsNone(row["mlm_guided_asr"])


if __name__ == "__main__":
    unittest.main()
