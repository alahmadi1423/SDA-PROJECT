"""Fake model client used by the offline tests."""
import json
from types import SimpleNamespace


class Msg:
    def __init__(self, content=None, tool_query=None):
        self.content = content
        self.tool_calls = None
        if tool_query:
            self.tool_calls = [SimpleNamespace(
                id="c1", function=SimpleNamespace(
                    name="search_docs", arguments=json.dumps({"query": tool_query})))]

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": "c1", "type": "function",
                                "function": {"name": "search_docs",
                                             "arguments": self.tool_calls[0].function.arguments}}]
        return d


def fake_client(*messages):
    """Returns replies in order and records every request in client.calls."""
    queue, calls = list(messages), []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=queue.pop(0))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def failing_client(exc):
    def create(**_):
        raise exc
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
