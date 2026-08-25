#!/usr/bin/env python3
"""Render the triage graph to an actual image, read from `Workflow.graph.edges`.

Reuses scripts/dump_graph.py's `collect()` rather than re-deriving node/edge
data a second way -- one source of truth for "what the graph actually is."
ADR-013's consequence extended to pictures: rename a node and this diagram
changes with it, because nothing here is hand-drawn.

Node/edge styling comes entirely from the live graph object:
  * agent nodes  (LlmAgent)         -> rounded box, labelled with model name
  * router nodes (2+ routed edges)  -> diamond
  * everything else (FunctionNode)  -> plain box
  * DEFAULT_ROUTE edges             -> dashed, red, explicitly labelled

GCP services are NOT part of `workflow.graph.edges` -- the ADK graph is only
the triage-service's internal workflow, so Pub/Sub, Firestore and Secret
Manager are added as a fixed annotation layer, sourced from
aksha_agent/infra/{slack,triage/triage_service}.py and the live topic/secret
names verified this session (gcloud pubsub topics list; SlackNotifier.
SECRET_IDS), not invented. They're visually distinct (dotted, grey) so it's
never ambiguous which part of the picture is graph-derived and which is
placed by hand.

    python3 scripts/render_diagram.py                        # writes docs/architecture.svg + .dot
    python3 scripts/render_diagram.py --out /tmp/arch.png --format png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import graphviz

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "docs" / "architecture"


def _load_dump_graph():
    spec = importlib.util.spec_from_file_location("dump_graph", REPO_ROOT / "scripts" / "dump_graph.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- styling --------------------------------------------------------------

FUNC_STYLE = {"shape": "box", "style": "filled,rounded", "fillcolor": "#EFEFEF", "fontname": "Helvetica"}
AGENT_STYLE = {"shape": "box", "style": "filled,rounded", "fillcolor": "#CFE8FF", "penwidth": "2", "fontname": "Helvetica-Bold"}
ROUTER_STYLE = {"shape": "diamond", "style": "filled", "fillcolor": "#FFE9B3", "fontname": "Helvetica-Bold"}
START_STYLE = {"shape": "circle", "style": "filled", "fillcolor": "#333333", "fontcolor": "white", "width": "0.5", "fixedsize": "true", "fontsize": "10"}

INFRA_STYLE = {"shape": "cylinder", "style": "filled", "fillcolor": "#E3E3E3", "fontname": "Helvetica-Oblique", "fontsize": "11"}
TOPIC_STYLE = {"shape": "component", "style": "filled", "fillcolor": "#D8F5D0", "fontname": "Helvetica-Oblique", "fontsize": "11"}
SERVICE_STYLE = {"fontname": "Helvetica-Bold", "fontsize": "13"}

DEFAULT_ROUTE_EDGE = {"style": "dashed", "color": "#B00020", "fontcolor": "#B00020", "penwidth": "1.5"}
ROUTED_EDGE = {"color": "#8A6D00", "fontcolor": "#8A6D00"}
PLAIN_EDGE = {"color": "#555555"}
INFRA_EDGE = {"style": "dotted", "color": "#777777", "fontcolor": "#777777", "fontsize": "10", "constraint": "false"}


def router_names(edges: list[dict]) -> set[str]:
    """A node is a router if any of its outgoing edges carries a route label."""
    return {e["from"] for e in edges if e["route"] is not None}


def build_graph(nodes: dict, edges: list, default_route_value) -> graphviz.Digraph:
    routers = router_names(edges)

    g = graphviz.Digraph("aksha_triage_architecture", format="svg")
    g.attr(rankdir="TB", fontname="Helvetica", labelloc="t", fontsize="20",
           label="AKSHA -- detect-to-route triage pipeline\n(generated from Workflow.graph.edges -- ADR-013)")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr(compound="true")

    # --- upstream: detector-service + telemetry-in, as a fixed annotation ---
    with g.subgraph(name="cluster_detector") as c:
        c.attr(label="detector-service (Cloud Run) -- fast path", style="dashed", color="#999999", **SERVICE_STYLE)
        c.node("telemetry_in", "Pub/Sub topic\ntelemetry-in", **TOPIC_STYLE)
        c.node("detect", "detect\n(IForest + split conformal)", **FUNC_STYLE)
        c.edge("telemetry_in", "detect")

    g.node("firestore", "Firestore\nincidents/ + traces/", **INFRA_STYLE)
    g.edge("detect", "firestore", label="DetectionResult", **INFRA_EDGE)

    with g.subgraph(name="cluster_triage") as t:
        t.attr(label="triage-service (Cloud Run) -- slow path, ADK 2 Workflow graph", style="solid", color="#666666", **SERVICE_STYLE)
        t.node("triage_topic", "Pub/Sub topic\ntriage", **TOPIC_STYLE)

        for node in sorted(nodes.values(), key=lambda n: n["name"]):
            name = node["name"]
            if name == "__START__":
                t.node(name, "START", **START_STYLE)
            elif node["is_agent"]:
                t.node(name, f"{name}\n({node['model']})", **AGENT_STYLE)
            elif name in routers:
                t.node(name, name, **ROUTER_STYLE)
            else:
                t.node(name, name, **FUNC_STYLE)

        t.edge("triage_topic", "__START__", label="push", **PLAIN_EDGE)

        for edge in edges:
            route = edge["route"]
            attrs = dict(PLAIN_EDGE)
            label = None
            if route is not None:
                if route == default_route_value:
                    label = "DEFAULT_ROUTE"
                    attrs = dict(DEFAULT_ROUTE_EDGE)
                else:
                    label = str(route)
                    attrs = dict(ROUTED_EDGE)
            t.edge(edge["from"], edge["to"], label=label, **attrs)

    g.edge("detect", "triage_topic", label="publish\nDetectionResult", **INFRA_EDGE)

    # --- Firestore also fed from inside the triage graph (every node, via
    # trace(); one final incidents/{id} write) -- one compound edge anchored
    # on a real node but drawn from the cluster boundary, not 15 edges from
    # every node.
    g.edge("verification_gate", "firestore", label="trace() every node\n+ final incident doc",
           ltail="cluster_triage", **INFRA_EDGE)

    # --- Secret Manager feeding the Slack notifier ---
    g.node("secret_manager", "Secret Manager\nslack-flight-director\nslack-subsystem\nslack-log", **INFRA_STYLE)
    for target in ("notify_flight_director", "notify_subsystem_engineer", "record_to_log", "record_unroutable"):
        g.edge("secret_manager", target, label="webhook URL", **INFRA_EDGE)

    return g


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output path stem (no extension)")
    parser.add_argument("--format", default="svg", choices=["svg", "png"])
    args = parser.parse_args()

    dg = _load_dump_graph()
    from google.adk.workflow import DEFAULT_ROUTE

    workflow = dg.build_workflow(
        detection=dg.PLACEHOLDER_DETECTION,
        incident_id="graph-dump",
        investigate_model="gemini-3.5-flash",
        explain_model="gemini-3.5-flash-lite",
    )
    nodes, edges = dg.collect(workflow)

    g = build_graph(nodes, edges, DEFAULT_ROUTE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.format = args.format
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(g.source)
    rendered = g.render(filename=str(out), cleanup=True)
    print(f"dot source : {dot_path}")
    print(f"rendered   : {rendered}")

    routers = router_names(edges)
    agents = [n for n in nodes.values() if n["is_agent"]]
    defaults = [e for e in edges if e["route"] == DEFAULT_ROUTE]
    print(f"\nnodes={len(nodes)} edges={len(edges)} agents={len(agents)} "
          f"routers={sorted(routers)} default_route_edges={len(defaults)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
