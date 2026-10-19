"""Offline tests: all API calls are fake; no credentials/network/GPU required."""
import argparse
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from paer import quality_relation_audit as q


def response(label="unchanged", **updates):
    content = {"relation_B_to_A": label, "confidence": "high", "affected_dimensions": [],
               "reason": "No substantial change.", "evidence_A": "", "evidence_B": ""}
    result = {"model": "deepseek-flash", "id": "fake", "system_fingerprint": "offline",
              "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    result.update(updates)
    return result


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.audit = self.root / "audit"
        self.valid = self.project / "data/valid_fold0.csv"
        self.valid.parent.mkdir(parents=True)
        with self.valid.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["essay_id", "full_text", "score", "prompt_name"])
            writer.writeheader()
            writer.writerows({"essay_id": str(i), "full_text": f"Original essay {i}.",
                             "score": i % 3 + 1, "prompt_name": f"topic{i % 2}"} for i in range(8))
        self.victims = {"b0": self.project / "deberta_checkpoints/fold0_best",
                        "mixed": self.project / "outputs/mixed/gstep1000",
                        "paer": self.project / "outputs/paer/gstep400"}
        for model, prefix in (("mixed", "aes_mixed_at_rhi_checkpoint_selection_uniform30_seed42"),
                              ("paer", "aes_paer_rhi_v3_checkpoint_selection_uniform30_seed42")):
            q.save_json(self.project / "outputs" / prefix / "best_checkpoint.json",
                        {"checkpoint_path": str(self.victims[model])})
        self.mapping = {}
        for model in q.MODELS:
            self.mapping[model] = {}
            for attack in q.ATTACKS:
                directory = self.project / "outputs" / f"eval_{model}" / attack
                details = [{"essay_id": str(i), "original_text": f"Original essay {i}.",
                            "perturbed_text": f"Modified essay {i}, {model}, {attack}.",
                            "original_score": 2.0, "perturbed_score": 2.12, "delta": 0.12}
                           for i in range(8)]
                path = directory / f"{attack}_details.json"
                q.save_json(path, {"summary": {"attack": attack}, "details": details})
                q.save_json(directory / "run_manifest.json", {
                    "victim": str(self.victims[model]), "data": str(self.valid),
                    "seed": 42, "n_essays": 8, "success_threshold": 0.1,
                    "attack_parameters": {"n_steps": 30, "beam_size": 1, "max_candidates_per_step": 16}})
                self.mapping[model][attack] = str(path)
        self.prepare_args = argparse.Namespace(project_root=self.project, output_dir=self.audit,
            seed=42, attack_seed=42, essays=4, sources_json=None, context_json=None)
        self.judge_args = argparse.Namespace(audit_dir=self.audit, max_pairs=1,
            max_tokens=512, timeout=30, allow_upload=True, dry_run=False)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            q.prepare(self.prepare_args)

    def judge(self, transport):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "FAKE_NOT_A_REAL_KEY"}), contextlib.redirect_stdout(io.StringIO()):
            q.judge(self.judge_args, transport=transport)

    def test_export_balanced_same_ids_and_blinding(self):
        self.prepare()
        manifest, records, pairs = q.verify_inputs(self.audit)
        self.assertEqual(manifest["n_records"], 36)
        self.assertEqual(len(pairs), 36)
        counts = Counter((r["model"], r["attack"]) for r in records)
        for model in q.MODELS:
            self.assertEqual([counts[(model, a)] for a in q.ATTACKS], [4, 4, 2, 2])
        before = (self.audit / "blind_pairs.json").read_bytes()
        self.prepare()
        self.assertEqual(before, (self.audit / "blind_pairs.json").read_bytes())
        body = q.request_body(pairs[0], "forward", 512)
        user_data = json.loads(body["messages"][1]["content"])
        self.assertEqual(set(user_data), {"context", "A", "B"})
        self.assertIsNone(user_data["context"]["task_prompt_text"])
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_exact_text_pair_dedup_and_source_preservation(self):
        for attack in q.ATTACKS:
            common = q.read_json(self.mapping["b0"][attack])
            for model in ("mixed", "paer"):
                q.save_json(self.mapping[model][attack], common)
        hashes = {p: q.file_hash(p) for m in self.mapping.values() for p in m.values()}
        self.prepare()
        self.assertEqual(len(q.read_json(self.audit / "blind_pairs.json")), 12)
        self.assertEqual(hashes, {p: q.file_hash(p) for p in hashes})

    def test_reject_different_duplicate_evaluation(self):
        path = Path(self.mapping["mixed"]["hotflip"])
        other = path.parent.parent / "duplicate_hotflip"
        payload = q.read_json(path)
        payload["details"][0]["perturbed_text"] += " Different."
        q.save_json(other / path.name, payload)
        q.save_json(other / "run_manifest.json", q.read_json(path.parent / "run_manifest.json"))
        with self.assertRaisesRegex(ValueError, "Different matching"):
            self.prepare()
        self.assertFalse(self.audit.exists())

    def test_explicit_sources_and_wrong_budget_rejected(self):
        mapping_path = self.root / "sources.json"
        q.save_json(mapping_path, self.mapping)
        self.prepare_args.sources_json = mapping_path
        manifest_path = Path(self.mapping["b0"]["hotflip"]).parent / "run_manifest.json"
        manifest = q.read_json(manifest_path)
        manifest["attack_parameters"]["n_steps"] = 10
        q.save_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "30 steps"):
            self.prepare()

    def test_parser_direction_and_invalid_evidence(self):
        pair = {"original_text": "Good essay.", "modified_text": "Bad essay."}
        text = response("improved")["choices"][0]["message"]["content"]
        self.assertEqual(q.parse_judgment(text, pair, "reverse")["normalized_relation"], "worsened")
        with self.assertRaises((ValueError, KeyError)):
            q.parse_judgment("{}", pair, "forward")
        data = json.loads(text)
        data["evidence_A"] = "Invented evidence"
        with self.assertRaisesRegex(ValueError, "Evidence"):
            q.parse_judgment(json.dumps(data), pair, "forward")

    def test_budget_reserve_resume_and_tamper(self):
        path = self.root / "budget.jsonl"
        ledger = q.BudgetLedger(path, 30)
        ledger.reserve("paid_or_unknown", 20)
        resumed = q.BudgetLedger(path, 30)
        with self.assertRaises(q.BudgetStop):
            resumed.reserve("new", 11)
        resumed.settle("paid_or_unknown", 2)
        resumed.reserve("new", 28)
        self.assertEqual(resumed.spent(), Decimal(30))
        path.write_text(path.read_text().replace('"28.000000"', '"0.000000"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            q.BudgetLedger(path, 30)

    def test_dry_run_never_network_or_create_binding(self):
        self.prepare()
        self.judge_args.dry_run = True
        with patch.object(q, "call_deepseek", side_effect=AssertionError("NO NETWORK")), contextlib.redirect_stdout(io.StringIO()):
            q.judge(self.judge_args)
        self.assertFalse((self.audit / "judge_binding.json").exists())
        self.assertFalse((self.audit / "budget_ledger.jsonl").exists())

    def test_permission_and_key_are_required(self):
        self.prepare()
        self.judge_args.allow_upload = False
        with self.assertRaisesRegex(ValueError, "API disabled"):
            q.judge(self.judge_args)
        self.judge_args.allow_upload = True
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "locally"):
                q.judge(self.judge_args)

    def test_cached_votes_resume_no_new_calls_and_summary(self):
        self.prepare()
        calls = []
        def fake(*args):
            calls.append(args[0])
            return response()
        self.judge(fake)
        self.assertEqual(len(calls), 2)
        self.judge(fake)
        self.assertEqual(len(calls), 2)
        output = self.root / "summary"
        with contextlib.redirect_stdout(io.StringIO()):
            q.summarize(argparse.Namespace(audit_dir=self.audit, output_dir=output))
        payload = q.read_json(output / "quality_relation_results.json")
        self.assertEqual(payload["unique_decision_status_counts"]["order_consistent"], 1)
        self.assertEqual(sum(r["n_resolved"] for r in payload["rows"]), 1)
        self.assertNotIn("FAKE_NOT_A_REAL_KEY", "".join(p.read_text(encoding="utf-8") for p in self.audit.rglob("*.json")))

    def test_conflict_third_vote_and_normalization(self):
        self.prepare()
        labels = iter(["improved", "improved", "improved"])
        self.judge(lambda *args: response(next(labels)))
        pair = q.read_json(self.audit / "blind_pairs.json")[0]
        binding = q.fingerprint(q.read_json(self.audit / "judge_binding.json"))
        votes = {o: q.load_vote(self.audit / "votes" / f"{pair['pair_id']}_{o}.json", binding, pair, o)
                 for o in ("forward", "reverse", "forward_repeat")}
        decision = q.final_relation(pair, votes)
        self.assertEqual(decision["status"], "majority_after_conflict")
        self.assertEqual(decision["relation"], "improved")

    def test_malformed_response_not_tie_and_failed_retry_cap(self):
        self.prepare()
        calls = []
        def fake(*args):
            calls.append(1)
            return response(choices=[{"message": {"content": "not json"}, "finish_reason": "stop"}])
        self.judge(fake)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(list((self.audit / "attempts").glob("*.json"))), 4)
        self.judge(fake)
        self.assertEqual(len(calls), 4)
        pair = q.read_json(self.audit / "blind_pairs.json")[0]
        self.assertIsNone(q.final_relation(pair, {})["relation"])

    def test_exhausted_budget_stops_before_network(self):
        self.prepare()
        q.BudgetLedger(self.audit / "budget_ledger.jsonl", 30).reserve("previous_unknown", 30)
        self.judge(lambda *args: self.fail("Must not call API"))
        self.assertEqual(q.read_json(self.audit / "budget_status.json")["remaining_cny"], "0")

    def test_http_auth_failure_stops_and_keeps_reservation(self):
        self.prepare()
        def fake(*args):
            raise HTTPError("https://api.deepseek.com/chat/completions", 401, "Unauthorized", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            self.judge(fake)
        self.assertGreater(q.BudgetLedger(self.audit / "budget_ledger.jsonl", 30).spent(), 0)
        self.assertEqual(len(list((self.audit / "attempts").glob("*.json"))), 1)

    def test_input_tamper_and_missing_ledger_refuse_reset(self):
        self.prepare()
        self.judge(lambda *args: response())
        (self.audit / "budget_ledger.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "budget ledger is missing"):
            self.judge(lambda *args: self.fail("Must not call"))
        rows = q.read_json(self.audit / "private_pair_records.json")
        rows[0]["delta"] = 100
        q.save_json(self.audit / "private_pair_records.json", rows)
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.judge(lambda *args: self.fail("Must not call"))

    def test_fixed_key_can_resume_auth_failure_without_reset(self):
        self.prepare()
        def unauthorized(*args):
            raise HTTPError("https://api.deepseek.com/chat/completions", 401, "Unauthorized", {}, None)
        with self.assertRaises(RuntimeError):
            self.judge(unauthorized)
        reserved = q.BudgetLedger(self.audit / "budget_ledger.jsonl", 30).spent()
        calls = []
        def fixed_key(*args):
            calls.append(1)
            return response()
        self.judge(fixed_key)
        self.assertEqual(len(calls), 2)
        spent = q.BudgetLedger(self.audit / "budget_ledger.jsonl", 30).spent()
        self.assertEqual(spent, reserved + 2 * q.usage_cost(response()["usage"]))

    def test_identical_text_does_not_call_api(self):
        self.prepare()
        pairs = q.read_json(self.audit / "blind_pairs.json")
        pairs[0]["modified_text"] = pairs[0]["original_text"]
        q.save_json(self.audit / "blind_pairs.json", pairs)
        manifest = q.read_json(self.audit / "source_manifest.json")
        manifest["blind_pairs_sha256"] = q.fingerprint(pairs)
        q.save_json(self.audit / "source_manifest.json", manifest)
        self.judge(lambda *args: self.fail("Identical text must be free"))
        self.assertFalse((self.audit / "budget_ledger.jsonl").exists())

    def test_missing_usage_retains_reserve_not_zero_charge(self):
        self.prepare()
        self.judge(lambda *args: response(usage={}))
        events = q.BudgetLedger(self.audit / "budget_ledger.jsonl", 30)
        self.assertEqual(len(events.events), 4)
        self.assertTrue(all(event["kind"] == "reserve" for event in events.events))
        self.assertGreater(events.spent(), 0)

    def test_vote_tamper_detected_before_more_calls(self):
        self.prepare()
        self.judge(lambda *args: response())
        path = next((self.audit / "votes").glob("*.json"))
        data = q.read_json(path)
        data["judgment"]["normalized_relation"] = "improved"
        q.save_json(path, data)
        with self.assertRaisesRegex(ValueError, "Cached judgment changed"):
            self.judge(lambda *args: self.fail("Must not call"))

    def test_three_distinct_votes_remain_conflict_not_unchanged(self):
        pair = {"original_text": "Before", "modified_text": "After"}
        votes = {order: {"status": "ok", "judgment": {"normalized_relation": label, "confidence": "high"}}
                 for order, label in zip(("forward", "reverse", "forward_repeat"), q.LABELS)}
        decision = q.final_relation(pair, votes)
        self.assertEqual(decision["status"], "unresolved_conflict")
        self.assertIsNone(decision["relation"])

    def test_os_lock_prevents_concurrent_budget_writers(self):
        self.audit.mkdir()
        with q.RunLock(self.audit):
            with self.assertRaisesRegex(RuntimeError, "Another audit process"):
                with q.RunLock(self.audit):
                    self.fail("Second process lock must not succeed")


if __name__ == "__main__":
    unittest.main()
