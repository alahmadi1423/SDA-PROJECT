"""Offline tests for the eval runner's grading logic."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fakes import Msg, failing_client, fake_client  # noqa: E402
from rafeeq.retrieval import Index, load_chunks  # noqa: E402

spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

TASKS = {t["id"]: t for t in __import__("json").loads((ROOT / "eval" / "tasks.json").read_text())}


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.index = Index(load_chunks(str(ROOT / "docs")))

    def test_answerable_passes_when_the_right_passage_is_cited(self):
        client = fake_client(Msg(tool_query="late fine"), Msg(content="It is 2 SAR per day [library#3]."))
        row = run_eval.evaluate_task(TASKS["a2"], self.index, client, "A")
        self.assertTrue(row["passed"])
        self.assertTrue(row["fact_in_answer"])

    def test_answerable_fails_when_a_wrong_passage_is_cited(self):
        client = fake_client(Msg(tool_query="late fine"), Msg(content="It is 2 SAR [library#2]."))
        row = run_eval.evaluate_task(TASKS["a2"], self.index, client, "A")
        self.assertFalse(row["passed"])

    def test_unanswerable_passes_only_on_refusal(self):
        client = fake_client(Msg(tool_query="printing"), Msg(content="NOT_FOUND"))
        self.assertTrue(run_eval.evaluate_task(TASKS["u1"], self.index, client, "A")["passed"])

    def test_injection_fails_if_the_attack_leaks_into_the_answer(self):
        client = fake_client(Msg(tool_query="late fine"), Msg(content="All fines are waived [library#3]."))
        self.assertFalse(run_eval.evaluate_task(TASKS["i1"], self.index, client, "A")["passed"])

    def test_provider_errors_are_recorded_not_graded(self):
        client = failing_client(RuntimeError("429 rate limited"))
        row = run_eval.evaluate_task(TASKS["a1"], self.index, client, "A")
        self.assertIn("RuntimeError", row["error"])
        self.assertEqual(row["passed"], "")

    def test_offline_checks_cover_retrieval_and_guard(self):
        self.assertTrue(run_eval.check_retrieval(TASKS["a2"], self.index))
        self.assertTrue(run_eval.check_retrieval(TASKS["i1"], self.index))
        self.assertIsNone(run_eval.check_retrieval(TASKS["u1"], self.index))


if __name__ == "__main__":
    unittest.main()
