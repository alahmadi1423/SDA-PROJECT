"""Human-approved report export.

The model has no tool for saving anything. Only this module writes files, and only after the
person has read the exact report and typed "y". Every decision is appended to an audit log.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .agent import Answer


def build_report(question: str, ans: Answer) -> dict:
    return {
        "question": question,
        "answer": ans.text,
        "citations": [{"id": c.id, "source": c.source, "heading": c.heading, "text": c.text}
                      for c in ans.citations],
        "quarantined_passages": ans.quarantined,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def render(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)


def _audit(out_dir: Path, decision: str, sha: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "decision": decision, "sha256": sha}
    with open(out_dir / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_with_approval(question: str, ans: Answer, out_dir: str = "reports",
                       confirm=input, echo=print) -> Path | None:
    """Show the exact report, ask for approval, and save only on an explicit yes."""
    if ans.refused:
        echo("Nothing to save: the answer was a refusal.")
        return None
    text = render(build_report(question, ans))
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    echo("\n--- Report preview ---\n" + text + "\n----------------------")
    reply = confirm("Approve and save this report? [y/N]: ").strip().lower()
    out = Path(out_dir)
    if reply not in {"y", "yes"}:
        _audit(out, "rejected", sha)
        echo("Rejected - no report was saved.")
        return None
    path = out / f"report-{sha[:12]}.json"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _audit(out, "approved", sha)
    echo(f"Saved {path}")
    return path
