#!/usr/bin/env python3
"""Offline evaluation of the triage graph's discrimination.

Runs the real graph -- both LLM nodes, the real context provider, the real
routers -- directly in-process over a stratified sample of the test split. No
Pub/Sub, no Cloud Run, no Firestore: this measures the graph's judgement, not
the delivery path.

The question it answers: given a window the detector flagged, does the verifier
distinguish a genuine fault from unusual-but-expected operation? A verifier that
confirms everything is worth exactly as much as no verifier at all, and the only
way to know which one we have is to run it against labelled windows and count.

    python3 scripts/eval_triage.py                      # default sample
    python3 scripts/eval_triage.py --per-class 10       # cheaper run
    python3 scripts/eval_triage.py --dry-run            # sample only, no LLM calls
    python3 scripts/eval_triage.py --golden             # the committed fixed set

Spend: 2 LLM calls per window. The default sample is ~62 windows, so ~124 calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PARQUET = "data/processed/mission2_features.parquet"
DEFAULT_GOLDEN = "tests/fixtures/golden_set.json"
CATEGORIES = ("nominal", "rare_event", "anomaly")
SEED = 20260818


def build_sample(frame: pd.DataFrame, per_class: int, artifact) -> pd.DataFrame:
    """Detector-flagged test windows, stratified by true category.

    Only flagged windows: an unflagged window never reaches triage in
    production, so scoring the verifier on one would measure something the
    system never does.
    """
    from aksha_core.detectors.iforest import feature_columns

    test = frame[frame["split"] == "test"].copy()
    scored = artifact.score_frame(test)
    test["score"] = scored["score"].to_numpy()
    test["conformal_p"] = scored["conformal_p"].to_numpy()
    test["threshold"] = scored["threshold"].to_numpy()
    flagged = test[test["score"] >= test["threshold"]]

    print("detector-flagged test windows by true category:")
    for category in CATEGORIES:
        print(f"  {category:<11} {int((flagged['label'] == category).sum()):>6,}")

    parts = []
    for category in CATEGORIES:
        rows = flagged[flagged["label"] == category]
        take = min(per_class, len(rows))
        if take:
            parts.append(rows.sample(n=take, random_state=SEED))
    sample = pd.concat(parts).reset_index(drop=True)
    sample.attrs["feature_columns"] = feature_columns(frame)
    return sample


async def run_one(row: pd.Series, columns: list[str], provider, semaphore) -> dict:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService

    from aksha_agent.graph.workflow import build_workflow

    detection = {
        "fragment_id": f"eval-{row.name}",
        "channel_id": row["channel"],
        "t_start": str(row["window_start"]),
        "t_end": str(row["window_start"] + pd.Timedelta(hours=1)),
        "score": float(row["score"]),
        "threshold": float(row["threshold"]),
        "conformal_p": float(row["conformal_p"]),
        "detector_version": "iforest-conformal-0.1.0",
        "features": {c: float(row[c]) for c in columns},
    }

    steps: list[tuple[str, dict]] = []
    workflow = build_workflow(
        detection=detection,
        incident_id=detection["fragment_id"],
        trace=lambda node, payload: steps.append((node, payload)),
        context_provider=provider,
    )

    result = {
        "true_label": row["label"],
        "channel": row["channel"],
        "conformal_p": detection["conformal_p"],
        "final_verdict": None,
        "gate_verdict": None,
        "llm_verdict": None,
        "gate_distance": None,
        "hypothesis": None,
        "severity": None,
        "tokens": 0,
        "thought_tokens": 0,
        "seconds": 0.0,
        "error": None,
    }

    async with semaphore:
        started = time.perf_counter()
        try:
            runner = Runner(
                node=workflow,
                app_name="eval",
                session_service=InMemorySessionService(),
                auto_create_session=True,
            )
            async for event in runner.run_async(
                user_id="eval", session_id=detection["fragment_id"], new_message=None
            ):
                meta = getattr(event, "usage_metadata", None)
                if meta is not None:
                    result["tokens"] += meta.total_token_count or 0
                    result["thought_tokens"] += meta.thoughts_token_count or 0
                out = getattr(event, "output", None)
                if isinstance(out, dict) and "routing_destination" in out:
                    result["final_verdict"] = out.get("final_verdict")
                    result["gate_verdict"] = out.get("gate_verdict")
                    result["llm_verdict"] = out.get("llm_verdict")
                    result["gate_distance"] = out.get("gate_distance")
                    result["severity"] = out.get("severity")
                    result["hypothesis"] = out.get("investigator_hypothesis")
                    result["reason"] = out.get("llm_reason")
        except Exception as exc:  # noqa: BLE001 - one bad window must not end the run
            result["error"] = f"{type(exc).__name__}: {exc}"[:200]
        result["seconds"] = round(time.perf_counter() - started, 2)

    for node, payload in steps:
        if node == "assemble_explainer_input":
            result["hypothesis"] = payload.get("hypothesis")
    return result


def confusion(results: list[dict]) -> str:
    statuses = ["confirm", "reject", "disputed", None]
    lines = [
        "",
        "CONFUSION: final verdict (columns) vs true category (rows)",
        "-" * 68,
        f"{'true category':<14}{'confirm':>10}{'reject':>10}{'disputed':>10}{'error':>10}{'n':>8}",
    ]
    for category in CATEGORIES:
        rows = [r for r in results if r["true_label"] == category]
        if not rows:
            continue
        counts = Counter(r["final_verdict"] if not r["error"] else None for r in rows)
        lines.append(
            f"{category:<14}"
            + "".join(f"{counts.get(s, 0):>10}" for s in statuses)
            + f"{len(rows):>8}"
        )
    return "\n".join(lines)


def three_way(results: list[dict]) -> str:
    """Gate, LLM and final side by side, so the override's effect is visible.

    Reporting only the final verdict would hide whether the gate or the model
    is carrying the result, which is the whole question this design exists to
    answer.
    """
    ok = [r for r in results if not r["error"]]
    lines = ["", "GATE vs LLM vs FINAL", "-" * 68]
    for source in ("gate_verdict", "llm_verdict", "final_verdict"):
        counts = Counter(r[source] for r in ok)
        label = {"gate_verdict": "gate", "llm_verdict": "llm", "final_verdict": "final"}[source]
        lines.append(f"  {label:<7} {dict(counts)}")

    # The invariant, checked on live data rather than only in the test suite:
    # if these ever diverge, the model has regained influence over the outcome.
    divergent = [r for r in ok if r["final_verdict"] != r["gate_verdict"]]
    lines.append(
        f"  final != gate: {len(divergent)}/{len(ok)}"
        + ("" if not divergent else "   <-- THE MODEL AFFECTED THE VERDICT")
    )

    agree = sum(1 for r in ok if r["gate_verdict"] == r["llm_verdict"])
    lines.append(
        f"  gate and llm agree: {agree}/{len(ok)} ({agree / max(len(ok),1):.0%})"
        "   [audit only -- disagreement changes nothing]"
    )

    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["true_label"], []).append(r)
    for category, rows in sorted(by_cat.items()):
        g = sum(1 for r in rows if r["gate_verdict"] == "confirm")
        l = sum(1 for r in rows if r["llm_verdict"] == "confirm")
        f = sum(1 for r in rows if r["final_verdict"] == "confirm")
        lines.append(
            f"    {category:<11} confirm-rate  gate {g}/{len(rows)}  "
            f"llm {l}/{len(rows)}  final {f}/{len(rows)}"
        )
    return "\n".join(lines)


def rates(results: list[dict]) -> str:
    """The two numbers that decide whether the verifier is worth its cost."""
    ok = [r for r in results if not r["error"]]
    faults = [r for r in ok if r["true_label"] == "anomaly"]
    expected = [r for r in ok if r["true_label"] in ("rare_event", "nominal")]

    lines = ["", "DISCRIMINATION", "-" * 68]
    if expected:
        rejected = sum(1 for r in expected if r["final_verdict"] == "reject")
        lines.append(
            f"  rejected as expected-operation, of non-anomaly windows : "
            f"{rejected}/{len(expected)} ({rejected / len(expected):.0%})"
        )
    if faults:
        confirmed = sum(1 for r in faults if r["final_verdict"] == "confirm")
        lines.append(
            f"  confirmed as fault, of true anomaly windows            : "
            f"{confirmed}/{len(faults)} ({confirmed / len(faults):.0%})"
        )
    by_status = Counter(r["final_verdict"] for r in ok)
    lines.append(f"  overall verdict mix: {dict(by_status)}")
    if len(by_status) == 1:
        lines.append(
            "  WARNING: the verifier returned one verdict for every window. It is "
            "not discriminating, whatever the accuracy looks like."
        )
    return "\n".join(lines)


def golden_sample(path: Path) -> pd.DataFrame:
    """Load the committed golden set as a scored frame.

    Everything the graph needs is already in the file -- features, detector
    score, conformal p, threshold -- so this never re-scores and never
    resamples. That is the point: two runs a week apart see the same windows
    with the same inputs, so a change in the numbers is a change in the design.
    """
    golden = json.loads(path.read_text())
    windows = golden["windows"]
    columns = list(windows[0]["features"])
    frame = pd.DataFrame(
        [
            {
                "channel": w["channel_id"],
                "window_start": pd.Timestamp(w["window_start"]),
                "label": w["true_label"],
                "score": w["score"],
                "conformal_p": w["conformal_p"],
                "threshold": w["detector_threshold"],
                "golden_group": w["group"],
                "golden_expected": w["expected_verdict"],
                **w["features"],
            }
            for w in windows
        ]
    )
    frame.attrs["feature_columns"] = columns
    frame.attrs["golden"] = golden
    return frame


def golden_report(results: list[dict], sample: pd.DataFrame) -> str:
    """Score against the expected verdicts, group by group.

    `ambiguous` rows are counted but never marked right or wrong -- they have no
    expected verdict on purpose. Reporting them as a hit rate would invent an
    answer the data does not have.
    """
    groups = sample["golden_group"].tolist()
    expected = sample["golden_expected"].tolist()
    lines = ["", "GOLDEN SET", "-" * 68,
             f"{'group':<16}{'n':>4}{'expected':>10}{'matched':>10}   verdicts"]
    hits = total = 0
    for group in ("clear_fault", "clear_expected", "clear_nominal", "ambiguous"):
        rows = [(r, e) for r, g, e in zip(results, groups, expected) if g == group]
        if not rows:
            continue
        mix = dict(Counter(r["final_verdict"] for r, _ in rows))
        want = rows[0][1]
        # pandas coerces the ambiguous group's None to NaN on the way through
        # the frame, and NaN is not None -- checking identity alone would score
        # rows that were deliberately given no expected verdict.
        if want is None or pd.isna(want):
            lines.append(f"{group:<16}{len(rows):>4}{'(none)':>10}{'-':>10}   {mix}")
            continue
        matched = sum(1 for r, e in rows if r["final_verdict"] == e)
        hits += matched
        total += len(rows)
        lines.append(f"{group:<16}{len(rows):>4}{want:>10}{f'{matched}/{len(rows)}':>10}   {mix}")
    if total:
        lines.append(f"\n  scored agreement (ambiguous excluded): {hits}/{total} "
                     f"({hits / total:.0%})")
    built = sample.attrs["golden"]
    lines.append(f"  built at gate threshold {built['gate_threshold_at_build']}, "
                 f"ambiguous band {built['ambiguous_band']}")
    return "\n".join(lines)


async def main_async(args) -> int:
    os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "True")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

    from aksha_agent.graph.context import ReferenceContextProvider
    from aksha_core.detectors.artifact import DetectorArtifact

    frame = pd.read_parquet(args.parquet)
    artifact = DetectorArtifact.load()

    if args.golden:
        sample = golden_sample(Path(args.golden_path))
        print(f"GOLDEN SET run: {len(sample)} fixed windows "
              f"({dict(Counter(sample['golden_group']))})")
    elif args.holdout:
        # STEP 6: fault sensitivity on anomaly windows absent from the reference.
        # Train-period, so not a headline metric -- but it turns n=1 into a real
        # measurement of whether the verifier recognises faults at all.
        from aksha_core.detectors.iforest import feature_columns

        sample = pd.read_parquet(args.holdout)
        scored = artifact.score_frame(sample)
        sample = sample.assign(
            score=scored["score"].to_numpy(),
            conformal_p=scored["conformal_p"].to_numpy(),
            threshold=scored["threshold"].to_numpy(),
        )
        if args.limit:
            sample = sample.head(args.limit)
        sample = sample.reset_index(drop=True)
        sample.attrs["feature_columns"] = feature_columns(frame)
        print(f"HOLDOUT fault-sensitivity run: {len(sample)} anomaly windows")
        print(f"  detector-flagged among them: "
              f"{int((sample['score'] >= sample['threshold']).sum())}")
    else:
        sample = build_sample(frame, args.per_class, artifact)
    columns = sample.attrs["feature_columns"]

    print(f"\nsample: {len(sample)} windows")
    print(sample["label"].value_counts().to_string())

    provider = ReferenceContextProvider()
    print(
        f"\ncontext reference: {len(provider.windows)} exemplars, "
        f"categories={provider.categories}, train_cut={provider.train_cut}"
    )
    if args.dry_run:
        print("\n--dry-run: no LLM calls made")
        return 0

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(run_one(row, columns, provider, semaphore) for _, row in sample.iterrows())
    )
    elapsed = time.perf_counter() - started

    print(confusion(results))
    print(three_way(results))
    print(rates(results))
    if args.golden:
        print(golden_report(results, sample))

    errors = [r for r in results if r["error"]]
    total_tokens = sum(r["tokens"] for r in results)
    thought_tokens = sum(r["thought_tokens"] for r in results)
    ok = len(results) - len(errors)
    print("")
    print("COST", "-" * 64, sep="\n")
    print(f"  windows           : {len(results)}  ({ok} ok, {len(errors)} errored)")
    print(f"  LLM calls         : {ok * 2}")
    print(f"  total tokens      : {total_tokens:,}  (thoughts {thought_tokens:,})")
    if ok:
        print(f"  tokens/incident   : {total_tokens // ok:,}")
    print(f"  wall time         : {elapsed:.1f}s at concurrency {args.concurrency}")
    if errors:
        print(f"\n  first errors: {[e['error'] for e in errors[:3]]}")

    hypotheses = Counter(r["hypothesis"] for r in results if r["hypothesis"])
    print(f"\nINVESTIGATOR HYPOTHESES\n{'-' * 68}")
    for name, count in hypotheses.most_common():
        print(f"  {name:<22} {count:>4}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nper-window results: {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--holdout", default=None,
                        help="parquet of held-out anomaly windows (STEP 6 fault sensitivity)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--golden", action="store_true",
                        help="evaluate the committed fixed golden set (comparable across runs)")
    parser.add_argument("--golden-path", default=DEFAULT_GOLDEN)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
