"""docs/EVALUATION.md's numeric claims must trace to eval/outputs/final_report.json.

This exists for the same reason tests/test_deploy_staging.py exists: a doc that
states a number can drift from the run that produced it the moment either one
is edited alone, and nothing else catches it -- same class of bug as the
deploy script shipping without the calibration artifact it needed.

Skipped when final_report.json is absent: it is gitignored and only produced
by a live scripts/eval_final.py run (which itself requires the golden-set and
holdout scripts/eval_triage.py passes -- real Gemini calls, not something CI
should trigger). When present, every percentage, a/b ratio, and 0.xxx decimal
in the RESULTS sections of the doc (section 1 through the end of section 6 --
the actual measured claims) must equal some value reachable in the JSON.

The "What this is not claiming" preamble is deliberately excluded: it cites
figures from other docs (e.g. OPSSAT-AD's 1594/529 split, ADR-006's cost
figures) that were never computed by eval_final.py and have no reason to
appear in final_report.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "EVALUATION.md"
REPORT = REPO / "eval" / "outputs" / "final_report.json"

SECTION_START = "## 1. Detector performance"
SECTION_END = "## Sources"

# Substrings that look like numeric claims but aren't -- version strings and
# citation ids, stripped before extraction rather than special-cased in the
# regex.
NOISE_PATTERNS = [
    r"arXiv \d{4}\.\d{5}",
    r"gemini-3\.5-flash(?:-lite)?",
    r"iforest-conformal-\d+\.\d+\.\d+",
]


def results_section_text() -> str:
    text = DOC.read_text()
    start = text.index(SECTION_START)
    end = text.index(SECTION_END)
    section = text[start:end]
    for pattern in NOISE_PATTERNS:
        section = re.sub(pattern, "", section)
    return section


def claimed_numbers(text: str) -> list[float]:
    text = text.replace("−", "-")  # typographic minus, e.g. a negative score in a table
    percents = [float(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]
    ratios: list[float] = []
    for num, den in re.findall(r"(\d+)/(\d+)", text):
        ratios += [float(num), float(den)]
    decimals = [
        float(f"{sign}{digits}")
        for sign, digits in re.findall(r"(-?)\b(0\.\d{2,4})\b", text)
    ]
    return percents + ratios + decimals


def flatten(value, out: list[float]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            flatten(v, out)
    elif isinstance(value, list):
        for v in value:
            flatten(v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))


def report_values() -> set[float]:
    raw: list[float] = []
    flatten(json.loads(REPORT.read_text()), raw)
    rounded: set[float] = set()
    for v in raw:
        for ndigits in (0, 1, 2, 3):
            rounded.add(round(v, ndigits))
    return rounded


@pytest.mark.skipif(
    not REPORT.exists(),
    reason="eval/outputs/final_report.json is gitignored; run scripts/eval_final.py first",
)
def test_evaluation_md_results_trace_to_final_report():
    known = report_values()
    missing = sorted(
        {n for n in claimed_numbers(results_section_text()) if round(n, 3) not in known}
    )
    assert not missing, (
        f"docs/EVALUATION.md cites numbers with no match in final_report.json: {missing} "
        "-- either the doc drifted from the data, or eval_final.py needs to report this value"
    )
