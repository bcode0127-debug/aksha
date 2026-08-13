# AKSHA — build rules

## Core/agent isolation

`aksha_core` imports **zero** of `google`, `adk`, `vertexai`. This is a hard boundary — the
detection/conformal core must be runnable and testable with no Google/ADK/Vertex dependency in
the import graph. CI enforces this with a grep over `aksha_core/` for those import roots; a
violation fails the build. If Google-specific glue is genuinely needed, it belongs in
`aksha_agent`, not `aksha_core`.

## Commits

Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `eval:`, ...). Subject line describes
the change, not the ticket.

## Workflow

- Read-remote-first: fetch/pull before starting new work on a branch.
- Never push without explicit OK from the user, even to a feature branch.
- Branch model: `main` is merge-only (no direct commits). One branch per task, named
  `chore/`, `feat/`, `fix/`, `docs/`, or `eval/` + short-desc. Squash-merge into `main`, then
  delete the branch.

## Dependencies

- `google-adk` is pinned at `2.3.0`. Re-verify all ADK-dependent code paths any time this pin is
  bumped — minor versions have changed agent/tool call signatures before.
- `ADK_MODEL` must be Gemini 3.5 or newer. The ADK tutorial defaults (older Gemini models) fail
  Stage One of the pipeline — do not use them, even for local smoke tests.

## Naming

Node names in `aksha_agent/graph` are a routing contract, not cosmetic labels — other code and
docs reference them by exact string. If you rename a node, grep all of `docs/` and the rest of
the repo for the old name and update every reference in the same commit.

## Reporting results

- Never report point-adjusted F1 as a standalone metric. It is known to overstate detector
  performance on this class of time-series anomaly task; if it's reported at all, report it
  alongside the non-adjusted metric with that caveat stated explicitly.
- No claims of standards compliance (e.g. ECSS, CCSDS) without citing the specific document
  number and a link to the source. "Standards-aligned" with no citation is not acceptable.
- No unverified numbers or citations in any file. Verify against source in-session or mark
  TODO. Looked-plausible is not verified.
