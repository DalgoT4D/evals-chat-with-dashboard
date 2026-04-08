"""
Eval suite for chat-with-dashboard using DeepEval.

Run with:
    uv run python evals.py

Scores each question on:
    1. Intent correctness — did it classify the query correctly?
    2. Table selection    — did the SQL touch the right tables?
    3. SQL correctness    — is the generated SQL semantically equivalent to expected?
    4. Answer quality     — does the answer meet the stated expectations?
"""

import json
import logging
import os
from datetime import datetime

from deepeval import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from auth import login
from client import run_single_question
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

intent_correctness = GEval(
    name="Intent Correctness",
    criteria=(
        "Check if the actual intent matches the expected intent. "
        "The actual_output contains '[intent=X]' and the expected_output contains '[expected_intent=Y]'. "
        "Score 1.0 if X == Y, score 0.0 otherwise."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)

table_selection = GEval(
    name="Table Selection",
    criteria=(
        "Check if the SQL query in actual_output references the correct tables. "
        "The expected_output lists the expected tables under '[expected_tables=...]'. "
        "Score 1.0 if the SQL touches exactly those tables (and no others from a different domain). "
        "Score 0.5 if it touches the right tables but also includes unnecessary ones. "
        "Score 0.0 if it misses required tables. "
        "If expected_tables is empty (non-SQL question), score 1.0 if no SQL was generated."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)

sql_correctness = GEval(
    name="SQL Correctness",
    criteria=(
        "Compare the generated SQL in actual_output against the expected SQL in expected_output. "
        "They do NOT need to be identical — they need to be semantically equivalent "
        "(same result set for the same data). "
        "Consider: column aliases, table aliases, JOIN order, WHERE clause equivalence, "
        "GROUP BY / ORDER BY equivalence. "
        "Score 1.0 if semantically equivalent. "
        "Score 0.5 if mostly correct but with minor differences (e.g. missing ORDER BY). "
        "Score 0.0 if fundamentally different logic. "
        "If expected_sql is null (non-SQL question), score 1.0 if no SQL was generated."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)

answer_quality = GEval(
    name="Answer Quality",
    criteria=(
        "Evaluate whether the answer in actual_output meets the expectations stated in expected_output "
        "under '[answer_expectations=...]'. "
        "Also consider the table descriptions provided for context on what the data represents. "
        "Score 1.0 if the answer fully meets expectations. "
        "Score 0.5 if partially met. "
        "Score 0.0 if expectations are not met at all."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)

ALL_METRICS = [intent_correctness, table_selection, sql_correctness, answer_quality]

# ---------------------------------------------------------------------------
# Build test cases by querying the live system
# ---------------------------------------------------------------------------


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path: str = "datasets/sample.json") -> list[dict]:
    path = os.path.join(_SCRIPT_DIR, path)
    logger.info("Loading dataset from %s", path)
    with open(path) as f:
        dataset = json.load(f)
    logger.info("Loaded %d eval questions", len(dataset))
    return dataset


def build_test_cases(
    cfg: Config, cookies: dict, dataset: list[dict]
) -> tuple[list[LLMTestCase], list[dict]]:
    """Send each question to the live system and build DeepEval test cases.

    Returns (test_cases, raw_responses) where raw_responses holds the
    original question, actual/expected values, and response latency.
    """
    test_cases = []
    raw_responses = []

    for i, item in enumerate(dataset):
        question = item["question"]
        logger.info("Querying question %d/%d: %s", i + 1, len(dataset), question)
        response = run_single_question(cfg, cookies, question)

        payload = response.get("payload", {})
        actual_intent = payload.get("intent", "unknown")
        actual_answer = response.get("content", "")
        actual_sql = payload.get("sql") or "NO SQL GENERATED"

        raw_responses.append({
            "question": question,
            "actual_intent": actual_intent,
            "actual_sql": actual_sql if actual_sql != "NO SQL GENERATED" else None,
            "actual_answer": actual_answer,
            "expected_intent": item["expected_intent"],
            "expected_tables": item.get("expected_tables", []),
            "table_descriptions": item.get("table_descriptions", {}),
            "expected_sql": item.get("expected_sql"),
            "answer_expectations": item.get("answer_expectations"),
            "response_latency_ms": response.get("response_latency_ms"),
        })

        actual_output = (
            f"[intent={actual_intent}]\n"
            f"[sql={actual_sql}]\n"
            f"[answer={actual_answer}]"
        )

        expected_output = (
            f"[expected_intent={item['expected_intent']}]\n"
            f"[expected_tables={json.dumps(item.get('expected_tables', []))}]\n"
            f"[table_descriptions={json.dumps(item.get('table_descriptions', {}))}]\n"
            f"[expected_sql={item.get('expected_sql') or 'NO SQL EXPECTED'}]\n"
            f"[answer_expectations={item.get('answer_expectations', 'No specific expectations')}]"
        )

        tc = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        test_cases.append(tc)

    return test_cases, raw_responses


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(cfg: Config, raw_responses: list[dict], eval_result: EvaluationResult) -> str:
    """Merge DeepEval scores into raw responses and write to results/."""
    results_dir = os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"eval_{timestamp}.json")

    # Index eval results by input text since evaluate() may reorder them
    scores_by_input = {}
    for tr in eval_result.test_results:
        scores_by_input[tr.input] = {
            md.name: {
                "score": md.score,
                "passed": md.success,
                "reason": md.reason,
            }
            for md in (tr.metrics_data or [])
        }

    results = []
    for raw in raw_responses:
        results.append({
            "question": raw["question"],
            "scores": scores_by_input.get(raw["question"], {}),
            "response_payload": {
                "actual_intent": raw["actual_intent"],
                "actual_sql": raw["actual_sql"],
                "actual_answer": raw["actual_answer"],
                "response_latency_ms": raw["response_latency_ms"],
            },
            "expected_intent": raw["expected_intent"],
            "expected_tables": raw["expected_tables"],
            "table_descriptions": raw.get("table_descriptions", {}),
            "expected_sql": raw["expected_sql"],
            "answer_expectations": raw["answer_expectations"],
        })

    output = {
        "run_at": timestamp,
        "environment": cfg.HTTP_BASE_URL,
        "dashboard_id": cfg.DASHBOARD_ID,
        "results": results,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = Config()
    cookies = login(cfg)

    dataset = load_dataset()

    logger.info("Querying live system...")
    test_cases, raw_responses = build_test_cases(cfg, cookies, dataset)
    logger.info("Got %d responses", len(test_cases))

    logger.info("Running DeepEval scoring...")
    result = evaluate(test_cases=test_cases, metrics=ALL_METRICS)

    save_results(cfg, raw_responses, result)


if __name__ == "__main__":
    main()
