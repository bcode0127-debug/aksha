"""The deploy script must ship every artifact the graph reads at runtime.

This exists because it did not. The verification gate loads its operating
threshold and ambiguous band from `mission2_recognition_calibration.json`, and
`scripts/deploy_triage.sh` staged only the context reference. On Cloud Run both
values would have come back None and the gate raises — a 500 per message, five
delivery attempts, then the dead-letter queue. The local test suite could not
see it, because locally the file is simply there.

So the check is on the SCRIPT, not on the running service: the failure mode is
"someone added an artifact and forgot the deploy script", and the only place
that is catchable before a deploy is here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEPLOY = REPO / "scripts" / "deploy_triage.sh"
CONTEXT_MODULE = REPO / "aksha_agent" / "graph" / "context.py"
ARTIFACTS = REPO / "aksha_core" / "artifacts"


def artifacts_the_graph_reads() -> set[str]:
    """Artifact filenames referenced as defaults by the graph's context module.

    Derived from the source rather than hardcoded here, so adding a third
    artifact makes this test start demanding it without anyone editing the
    test.
    """
    source = CONTEXT_MODULE.read_text()
    return set(re.findall(r'_ARTIFACTS / "([^"]+\.json)"', source))


def test_the_graph_reads_at_least_the_two_known_artifacts():
    """Guard the guard: if the regex above stops matching, every other test in
    this file would pass vacuously.
    """
    found = artifacts_the_graph_reads()
    assert {
        "mission2_context_reference.json",
        "mission2_recognition_calibration.json",
    } <= found, found


@pytest.mark.parametrize("name", sorted(artifacts_the_graph_reads()))
def test_deploy_script_stages_every_artifact_the_graph_reads(name):
    assert name in DEPLOY.read_text(), (
        f"{name} is read by the graph but never staged by deploy_triage.sh; "
        "the deployed service would not find it"
    )


@pytest.mark.parametrize("name", sorted(artifacts_the_graph_reads()))
def test_deploy_script_fails_fast_when_an_artifact_is_missing(name):
    """A missing artifact must stop the deploy, not produce a broken service.

    Silently deploying without it is the worse failure: the service comes up
    healthy, accepts messages, and 500s every one of them.
    """
    text = DEPLOY.read_text()
    assert "required artifact missing" in text
    assert name in text


@pytest.mark.parametrize("name", sorted(artifacts_the_graph_reads()))
def test_each_required_artifact_names_how_to_rebuild_it(name):
    """The error message has to say what to run, or it just relocates the
    problem to whoever hits it at 2am three days before a freeze.
    """
    line = next(l for l in DEPLOY.read_text().splitlines() if name in l and ":python3" in l)
    script = line.split(":python3", 1)[1].strip().strip('"').split()[-1]
    assert (REPO / script).exists(), f"{name} points at a missing builder: {script}"


@pytest.mark.parametrize("name", sorted(artifacts_the_graph_reads()))
def test_the_artifacts_are_committed(name):
    """They are loaded from the repo, so an uncommitted one is a deploy that
    works on this machine and nowhere else.
    """
    assert (ARTIFACTS / name).exists(), f"{name} missing from {ARTIFACTS}"
