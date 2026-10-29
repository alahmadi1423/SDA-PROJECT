"""Usage: python cli.py                          (interactive)
       python cli.py -q "your question"         (single question)
       python cli.py -q "..." --save            (ask for approval to save a report)"""
import argparse

from rafeeq.agent import PROMPTS, ask, make_client
from rafeeq.reports import save_with_approval
from rafeeq.retrieval import Index, load_chunks


def show(ans):
    print("\n" + ans.text)
    for c in ans.citations:
        print(f"  [{c.id}] {c.source} - {c.heading}")
    if ans.quarantined:
        print(f"  (quarantined suspicious passages: {', '.join(ans.quarantined)})")
    print()


def main():
    ap = argparse.ArgumentParser(description="Rafeeq - document-grounded assistant")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("-q", "--question")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="B", help="prompt version")
    ap.add_argument("--save", action="store_true", help="offer to save a report (needs approval)")
    ap.add_argument("--out", default="reports", help="folder for approved reports")
    args = ap.parse_args()

    index = Index(load_chunks(args.docs))
    client = make_client()

    def run(question):
        ans = ask(question, index, client, prompt=args.prompt)
        show(ans)
        if args.save:
            save_with_approval(question, ans, out_dir=args.out)

    if args.question:
        return run(args.question)
    while (q := input("Question (or 'exit'): ").strip()) and q != "exit":
        run(q)


if __name__ == "__main__":
    main()
