"""Offline tests for the agent loop: a fake client simulates a model that calls the search tool."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fakes import Msg, fake_client  # noqa: E402
from rafeeq.agent import PROMPTS, ask  # noqa: E402
from rafeeq.retrieval import Index, load_chunks  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.index = Index(load_chunks(str(DOCS)))

    def test_valid_citation_is_kept(self):
        client = fake_client(Msg(tool_query="late fine"),
                             Msg(content="The fine is 2 SAR per day [library#3]."))
        ans = ask("What is the late fine?", self.index, client)
        self.assertFalse(ans.refused)
        self.assertEqual([c.id for c in ans.citations], ["library#3"])

    def test_answer_without_valid_citation_is_refused(self):
        client = fake_client(Msg(tool_query="late fine"),
                             Msg(content="The fine is 100 SAR [library#99]."))
        self.assertTrue(ask("What is the late fine?", self.index, client).refused)

    def test_not_found_is_a_refusal(self):
        client = fake_client(Msg(tool_query="printing fees"), Msg(content="NOT_FOUND"))
        self.assertTrue(ask("What are the printing fees?", self.index, client).refused)

    def test_poisoned_chunk_is_quarantined(self):
        client = fake_client(Msg(tool_query="loan system maintenance"), Msg(content="NOT_FOUND"))
        ans = ask("When is maintenance?", self.index, client)
        self.assertEqual(ans.quarantined, ["notice#1"])

    def test_prompt_version_is_sent_to_the_model(self):
        for version in ("A", "B"):
            client = fake_client(Msg(content="NOT_FOUND"))
            ask("anything", self.index, client, prompt=version)
            self.assertEqual(client.calls[0]["messages"][0]["content"], PROMPTS[version])
        self.assertNotEqual(PROMPTS["A"], PROMPTS["B"])


if __name__ == "__main__":
    unittest.main()
