import json
from pathlib import Path

import pandas as pd

from app.schemas import AnalysisRequest
from app.services.analysis import analyze_dataframe


CASES_PATH = (
    Path(__file__).resolve().parents[1] / "evals" / "analysis_cases.json"
)


def test_synthetic_evaluation_cases_match_deterministic_analysis():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    for case in cases:
        request = AnalysisRequest.model_validate(case["expected_plan"])
        result = analyze_dataframe(pd.DataFrame(case["data"]), request)

        if "expected_rows" in case:
            assert result["table"]["rows"] == case["expected_rows"], case["id"]
        if "expected_trend" in case:
            insight = result["trend"][0]
            for key, value in case["expected_trend"].items():
                assert insight[key] == value, case["id"]
        if "expected_chart_type" in case:
            assert result["charts"][0]["type"] == case["expected_chart_type"], case["id"]
