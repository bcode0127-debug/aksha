#!/usr/bin/env python3
"""Print the triage graph's real structure, read from `Workflow.graph.edges`.

ADR-013's consequence: the architecture diagram is generated from the running
object, so the diagram is provably the system rather than a drawing that
drifted away from it. Nothing here is hand-maintained — rename a node and this
output changes with it.

    python3 scripts/dump_graph.py              # readable summary
    python3 scripts/dump_graph.py --mermaid    # mermaid flowchart
    python3 scripts/dump_graph.py --json       # machine-readable

Builds the workflow with placeholder models so it runs without credentials and
without making a single LLM call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from google.adk.workflow import DEFAULT_ROUTE  # noqa: E402

from aksha_agent.graph.workflow import build_workflow  # noqa: E402

PLACEHOLDER_DETECTION = {
    "fragment_id": "graph-dump",
    "channel_id": "channel_18",
    "t_start": "2001-12-14T19:00:00Z",
    "t_end": "2001-12-14T20:00:00Z",
    "score": 0.0,
    "threshold": 0.0,
    "conformal_p": 0.5,
    "detector_version": "graph-dump",
    "features": {},
}

AGENT_TYPES = {"LlmAgent", "Agent"}


def collect(workflow):
    """Nodes and edges as plain data, straight from the live graph object."""
    edges = []
    nodes: dict[str, dict] = {}

    for edge in workflow.graph.edges:
        for node in (edge.from_node, edge.to_node):
            kind = type(node).__name__
            nodes.setdefault(
                node.name,
                {
                    "name": node.name,
                    "kind": kind,
                    "is_agent": kind in AGENT_TYPES,
                    "model": getattr(node, "model", None),
                    "input_schema": getattr(
                        getattr(node, "input_schema", None), "__name__", None
                    ),
                    "output_schema": getattr(
                        getattr(node, "output_schema", None), "__name__", None
                    ),
                    "retry_config": (
                        None
                        if getattr(node, "retry_config", None) is None
                        else {
                            "max_attempts": node.retry_config.max_attempts,
                            "exceptions": node.retry_config.exceptions,
                        }
                    ),
                    "timeout": getattr(node, "timeout", None),
                },
            )
        edges.append(
            {"from": edge.from_node.name, "to": edge.to_node.name, "route": edge.route}
        )
    return nodes, edges


def render_text(nodes: dict, edges: list) -> str:
    agents = [n for n in nodes.values() if n["is_agent"]]
    lines = [
        "AKSHA triage graph",
        "=" * 60,
        f"nodes: {len(nodes)}   edges: {len(edges)}   agent nodes: {len(agents)}",
        f"LLM calls per incident: {len(agents)}",
        "",
        "NODES",
    ]
    for node in sorted(nodes.values(), key=lambda n: n["name"]):
        marker = "agent " if node["is_agent"] else "func  "
        line = f"  {marker} {node['name']}"
        if node["model"]:
            line += f"  model={node['model']}"
        if node["input_schema"]:
            line += f"  in={node['input_schema']}"
        if node["output_schema"]:
            line += f"  out={node['output_schema']}"
        lines.append(line)
        if node["retry_config"]:
            lines.append(
                f"          retry: max_attempts={node['retry_config']['max_attempts']} "
                f"exceptions={node['retry_config']['exceptions']}"
            )
    lines += ["", "EDGES"]
    for edge in edges:
        if edge["route"] is None:
            lines.append(f"  {edge['from']} -> {edge['to']}")
        else:
            tag = "DEFAULT_ROUTE" if edge["route"] == DEFAULT_ROUTE else repr(edge["route"])
            lines.append(f"  {edge['from']} -[{tag}]-> {edge['to']}")

    defaults = [e for e in edges if e["route"] == DEFAULT_ROUTE]
    lines += [
        "",
        f"DEFAULT_ROUTE edges: {len(defaults)}",
    ]
    for edge in defaults:
        lines.append(f"  {edge['from']} -> {edge['to']}")
    if not defaults:
        lines.append("  NONE — an unmatched route would end its branch silently (ADR-013)")
    return "\n".join(lines)


def render_mermaid(nodes: dict, edges: list) -> str:
    def node_id(name: str) -> str:
        return name.replace("__", "_")

    lines = ["flowchart TD"]
    for node in sorted(nodes.values(), key=lambda n: n["name"]):
        label = node["name"]
        if node["is_agent"]:
            lines.append(f'    {node_id(label)}["{label}<br/>{node["model"]}"]')
        else:
            lines.append(f'    {node_id(label)}("{label}")')
    for edge in edges:
        arrow = "-->"
        if edge["route"] is not None:
            tag = "DEFAULT" if edge["route"] == DEFAULT_ROUTE else edge["route"]
            arrow = f"--{tag}-->"
        lines.append(f"    {node_id(edge['from'])} {arrow} {node_id(edge['to'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mermaid", action="store_true")
    group.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workflow = build_workflow(
        detection=PLACEHOLDER_DETECTION,
        incident_id="graph-dump",
        investigate_model="gemini-3.5-flash",
        explain_model="gemini-3.5-flash-lite",
    )
    nodes, edges = collect(workflow)

    if args.json:
        print(json.dumps({"nodes": list(nodes.values()), "edges": edges}, indent=2))
    elif args.mermaid:
        print(render_mermaid(nodes, edges))
    else:
        print(render_text(nodes, edges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
