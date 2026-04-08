# Chat with Dashboard - Eval Harness

Accuracy evaluation harness for Dalgo's [chat-with-dashboard](https://github.com/DalgoT4D/DDP_backend) feature. Sends questions to a live Dalgo environment via WebSocket, captures responses, and scores them using [DeepEval](https://github.com/confident-ai/deepeval) with LLM-as-judge metrics.

## How it works

```
Golden dataset (JSON)
        |
        v
Login via /api/v2/login/ (cookie-based auth)
        |
        v
For each question:
    Connect via WebSocket -> Send question -> Wait for assistant_message
        |
        v
Collect all responses
        |
        v
DeepEval evaluate() scores each response using GPT-4.1 as judge
        |
        v
Save merged results (responses + scores) to results/
```

## Metrics

Each question is scored on 4 dimensions:

| Metric | What it checks | Scoring |
|---|---|---|
| **Intent Correctness** | Did the system classify the query correctly? (e.g. `query_with_sql`, `irrelevant`) | 1.0 = exact match, 0.0 = mismatch |
| **Table Selection** | Did the SQL touch the right tables? | 1.0 = exact tables, 0.5 = right + extra, 0.0 = missed tables |
| **SQL Correctness** | Is the generated SQL semantically equivalent to expected? | 1.0 = equivalent, 0.5 = minor differences, 0.0 = fundamentally different |
| **Answer Quality** | Does the answer meet stated expectations? | 1.0 = fully met, 0.5 = partially, 0.0 = not met |

Metrics use DeepEval's [GEval](https://docs.confident-ai.com/docs/metrics-llm-evals) which calls OpenAI (GPT-4.1) as the judge. Each metric makes 2 OpenAI calls per question — one to generate evaluation steps, one to score. Evaluation steps are cached after the first run, so subsequent runs make 1 call per metric per question.

```
First run OpenAI calls  = questions x metrics x 2
Cached run OpenAI calls = questions x metrics x 1
```

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in environment config
cp .env.example .env
```

### Environment variables

| Variable | Description | Example |
|---|---|---|
| `HTTP_BASE_URL` | Backend HTTP URL | `http://localhost:8002` |
| `WEBSOCKET_BASE_URL` | Backend WebSocket URL | `ws://localhost:8002` |
| `USERNAME` | Dalgo user email | `user@example.com` |
| `PASSWORD` | Dalgo user password | `password` |
| `ORG_SLUG` | Organization slug | `my-org` |
| `DASHBOARD_ID` | Dashboard ID to evaluate against | `65` |
| `OPENAI_API_KEY` | OpenAI API key (for DeepEval judge) | `sk-...` |

## Golden dataset format

Datasets live in `datasets/` as JSON files. Each entry:

```json
{
    "question": "How many beneficiaries are there in total?",
    "expected_intent": "query_with_sql",
    "expected_tables": ["fact_beneficiaries"],
    "table_descriptions": {
        "fact_beneficiaries": "Contains one row per beneficiary with demographic and enrollment data"
    },
    "expected_sql": "SELECT COUNT(*) FROM fact_beneficiaries",
    "answer_expectations": "Should return a single count number"
}
```

| Field | Required | Description |
|---|---|---|
| `question` | Yes | The question to ask the chat system |
| `expected_intent` | Yes | Expected intent classification (`query_with_sql`, `query_without_sql`, `follow_up_sql`, `follow_up_context`, `needs_clarification`, `small_talk`, `irrelevant`) |
| `expected_tables` | Yes | Tables the SQL should reference (empty list for non-SQL questions) |
| `table_descriptions` | Yes | Brief description of each expected table's data (helps the judge evaluate) |
| `expected_sql` | Yes | Expected SQL query (`null` for non-SQL questions) |
| `answer_expectations` | Yes | Free-text description of what the answer should contain |

## Running evals

```bash
uv run python evals.py
```

Results are saved to `results/eval_<timestamp>.json`.

## Results format

```json
{
    "run_at": "20260408_141051",
    "environment": "http://localhost:8002",
    "dashboard_id": 65,
    "results": [
        {
            "question": "How many beneficiaries are there in total?",
            "scores": {
                "Intent Correctness [GEval]": {
                    "score": 1.0,
                    "passed": true,
                    "reason": "..."
                },
                "Table Selection [GEval]": { ... },
                "SQL Correctness [GEval]": { ... },
                "Answer Quality [GEval]": { ... }
            },
            "response_payload": {
                "actual_intent": "query_with_sql",
                "actual_sql": "SELECT COUNT(*) FROM ...",
                "actual_answer": "There are 1234 beneficiaries...",
                "response_latency_ms": 11755
            },
            "expected_intent": "query_with_sql",
            "expected_tables": ["fact_beneficiaries"],
            "table_descriptions": { ... },
            "expected_sql": "SELECT COUNT(*) FROM fact_beneficiaries",
            "answer_expectations": "Should return a single count number"
        }
    ]
}
```

## Project structure

```
evals-chat-with-dashboard/
├── evals.py              # Main eval runner - builds test cases, runs scoring, saves results
├── auth.py               # Cookie-based login against Dalgo /api/v2/login/
├── client.py             # WebSocket client - sends questions, captures assistant_message responses
├── config.py             # Environment config from .env
├── pyproject.toml        # Dependencies (uv-managed)
├── datasets/
│   └── sample.json       # Golden dataset (template)
└── results/              # Eval results (gitignored)
```
