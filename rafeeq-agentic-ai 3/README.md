# Rafeeq - a document-grounded AI agent with citations and human approval

Rafeeq answers questions **only** from a folder of documents (Markdown/text), cites the passage behind every
answer, and refuses when the documents don't contain the answer. Nothing is saved unless a person reads the
exact report and approves it. It runs on free models through OpenRouter.

## How it works
```
question -> agent calls search_docs -> BM25 returns the best passages -> guard quarantines suspicious ones
         -> model writes an answer with [id] citations -> validator: every citation must come from the
            retrieved passages, otherwise the answer becomes a refusal
         -> (optional) report preview -> human approves or rejects -> saved report + audit entry
```

| File | Purpose |
|---|---|
| `rafeeq/retrieval.py` | Heading-aware chunking + BM25 ranking with light stemming |
| `rafeeq/guard.py` | Prompt-injection detection; retrieved text is wrapped as data |
| `rafeeq/agent.py` | Tool-calling loop, two prompt versions (A/B), model fallbacks, citation validation, refusals |
| `rafeeq/reports.py` | Human-approved report export and audit log |
| `eval/` | 12 evaluation tasks and a runner that compares prompts A and B (resumable, CSV output) |
| `tests/` | Offline unit tests using a fake model client |
| `.github/workflows/ci.yml` | Runs the tests and the offline eval on every push |

## Run it
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...          # free key from openrouter.ai
python cli.py -q "What is the late fine?"
python cli.py -q "What is the late fine?" --save     # asks for approval before saving a report
```
Optional: `export RAFEEQ_MODELS="model1,model2"` to change the model and fallback order; `--prompt A|B` to pick
the prompt version (default B).

## Human approval
The model has no tool for saving files. With `--save`, Rafeeq prints the exact report (question, answer,
cited passages, quarantined passages) and writes it to `reports/report-<sha256>.json` only if you type `y`.
Anything else, or a refusal answer, writes no report. Every decision is appended to `reports/audit.jsonl`.

## Evaluation
```bash
python eval/run_eval.py --retrieval-only    # offline: answer passage retrieved? poisoned passage flagged?
python eval/run_eval.py --prompt both       # full agent, prompt A vs B, appends to eval/results.csv
python eval/run_eval.py --prompt both --resume --delay 3   # continue after a quota stop
python -m unittest discover tests           # offline unit tests
```
Task types: **answerable** (must cite the passage containing the expected fact), **unanswerable** (must refuse),
**injection** (must not be influenced by an instruction hidden inside a document).
Prompt A is the baseline rules; prompt B adds guidance for multi-part questions, exact numbers and partial answers.
Provider errors (rate limits, outages) are recorded as errors and are not counted as wrong answers.
The CSV also records `fact_in_answer`, whether the expected fact appears in the answer text.

### Results
Fill this in from your own live run (`eval/results.csv`); do not report offline numbers as model accuracy.

| Prompt | Passed / ran | Provider errors | Model | Date |
|---|---|---|---|---|
| A | | | | |
| B | | | | |

## Make it yours
1. Replace the sample files in `docs/` with your own documents (`#`/`##` headings become passage boundaries).
2. Rewrite `eval/tasks.json` with questions about your documents. `expect_contains` is text that must appear in the correct passage.
3. Hide a malicious instruction in one of your documents and point `poisoned_contains` at it.

## Known limitations
- BM25 matches words, not meaning, so synonyms can be missed. Natural upgrade: embeddings or hybrid search.
- The offline retrieval check is weak on a tiny corpus (top-3 of a handful of passages); only the live run measures the agent.
- The guard is pattern-based: it quarantines the whole passage (even useful text in it) and won't catch every injection phrasing.
- Free OpenRouter models are rate-limited and some handle tool calling poorly, hence the fallback list and `--resume`.
- Approval is a local terminal prompt: there is no authentication and the audit log is a plain local file.
- Retrieval stemming is English-only.
