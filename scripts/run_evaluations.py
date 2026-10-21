import argparse
import json
from pathlib import Path

import pandas as pd

from app.graph.workflow import run_agent_analysis
from app.schemas import AnalysisRequest


CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "analysis_cases.json"


def _rows_match(actual: list[dict], expected: list[dict]) -> bool:
    normalized_actual = sorted(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) for row in actual)
    )
    normalized_expected = sorted(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) for row in expected)
    )
    return normalized_actual == normalized_expected


def run_evaluations() -> dict:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("Evaluation case set must not be empty.")
    reports = []
    for case in cases:
        report = {
            "id": case["id"],
            "expected_specialist": case["expected_specialist"],
            "expected_operation": case["expected_operation"],
            "routing_correct": False,
            "operation_correct": False,
            "plan_correct": False,
            "tool_succeeded": False,
            "output_correct": False,
        }
        try:
            specialist, plan, result = run_agent_analysis(
                case["question"],
                pd.DataFrame(case["data"]),
            )
            report["routing_correct"] = (
                specialist.value == case["expected_specialist"]
            )
            report["actual_specialist"] = specialist.value
            report["operation_correct"] = (
                plan.operation.value == case["expected_operation"]
            )
            report["actual_operation"] = plan.operation.value
            expected_plan = AnalysisRequest.model_validate(case["expected_plan"])
            actual_plan = plan.model_dump(mode="json")
            expected_plan_data = expected_plan.model_dump(mode="json")
            report["plan_correct"] = all(
                actual_plan.get(field) == expected_plan_data.get(field)
                for field in case["expected_plan"]
            )
            report["tool_succeeded"] = True
            output_correct = True
            if "expected_rows" in case:
                output_correct &= _rows_match(
                    result["table"]["rows"],
                    case["expected_rows"],
                )
            if "expected_trend" in case:
                insight = next(
                    (
                        item
                        for item in result.get("trend", [])
                        if item["metric"] == case["expected_trend"]["metric"]
                    ),
                    {},
                )
                output_correct &= all(
                    insight.get(key) == value
                    for key, value in case["expected_trend"].items()
                )
            if "expected_chart_type" in case:
                output_correct &= any(
                    chart["type"] == case["expected_chart_type"]
                    for chart in result.get("charts", [])
                )
            report["output_correct"] = bool(output_correct)
        except Exception as exc:
            report["error_type"] = type(exc).__name__
        reports.append(report)

    metrics = {
        key: (
            sum(bool(report[key]) for report in reports) / len(reports)
            if reports
            else 0.0
        )
        for key in (
            "routing_correct",
            "operation_correct",
            "plan_correct",
            "tool_succeeded",
            "output_correct",
        )
    }
    return {"case_count": len(reports), "metrics": metrics, "cases": reports}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic data-analysis LLM evaluation set."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any case or output check fails.",
    )
    args = parser.parse_args()
    report = run_evaluations()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and any(
        not all(
            case[key]
            for key in (
                "routing_correct",
                "operation_correct",
                "plan_correct",
                "tool_succeeded",
                "output_correct",
            )
        )
        for case in report["cases"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
