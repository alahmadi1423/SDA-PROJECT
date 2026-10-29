"""Offline tests for the human-approval gate."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rafeeq.agent import Answer  # noqa: E402
from rafeeq.reports import save_with_approval  # noqa: E402
from rafeeq.retrieval import Chunk  # noqa: E402

CHUNK = Chunk("library#3", "library.md", "Late fines", "The late fine is 2 SAR per day per book.")
GOOD = Answer("The fine is 2 SAR per day [library#3].", [CHUNK])
REFUSAL = Answer("I couldn't find an answer.", [], refused=True)


def audit_lines(folder):
    path = Path(folder) / "audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class ApprovalTests(unittest.TestCase):
    def test_approval_saves_report_named_by_its_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = save_with_approval("fine?", GOOD, d, confirm=lambda _: "y", echo=lambda *_: None)
            self.assertIsNotNone(path)
            data = path.read_text(encoding="utf-8")
            self.assertIn(hashlib.sha256(data.encode()).hexdigest()[:12], path.name)
            self.assertEqual(json.loads(data)["citations"][0]["id"], "library#3")
            self.assertEqual(audit_lines(d)[0]["decision"], "approved")

    def test_rejection_writes_no_report(self):
        with tempfile.TemporaryDirectory() as d:
            for reply in ("n", "", "maybe"):
                self.assertIsNone(save_with_approval(
                    "fine?", GOOD, d, confirm=lambda _, r=reply: r, echo=lambda *_: None))
            self.assertEqual(list(Path(d).glob("report-*.json")), [])
            self.assertTrue(all(e["decision"] == "rejected" for e in audit_lines(d)))

    def test_refusals_are_never_offered_for_saving(self):
        with tempfile.TemporaryDirectory() as d:
            asked = []
            result = save_with_approval("x", REFUSAL, d, confirm=lambda p: asked.append(p) or "y",
                                        echo=lambda *_: None)
            self.assertIsNone(result)
            self.assertEqual(asked, [])
            self.assertEqual(audit_lines(d), [])


if __name__ == "__main__":
    unittest.main()
