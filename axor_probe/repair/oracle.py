"""Wire the localization escape-oracle to the probe's structural readout.

``localize`` needs ``escapes(present) -> bool``: given the fragment ids kept in a
sandbox copy, does the agent escape? This builds that oracle from the probe's own
machinery — replay the present fragments + the probe scenario, run inference, read
the fields structurally, and compare against a fixed clean shadow via the residual.

Offline, so inference is a plain sync callable (``messages -> raw output``); the
caller adapts its async/HTTP client. ``clean_shadow`` is computed once (the agent on
the clean summary) and reused for every subset — the differential reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from axor_probe.comparator.residual import residual_payloads
from axor_probe.executor.readout import structural_readout
from axor_probe.executor.runner import ProbeResponse
from axor_probe.repair.localize import EscapeOracle

# Sync inference for the offline search: a list of message dicts → the model's raw
# output ({"text", "tool_calls", ...}). No async on the localization hot loop.
SyncInference = Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class OracleFragment:
    """A candidate context fragment as a replayable message."""
    fragment_id: str
    message: dict[str, Any]


def make_escape_oracle(
    *,
    fragments: list[OracleFragment],
    probe_messages: list[dict[str, Any]],   # appended after the context (canary + scenario)
    run_inference: SyncInference,
    clean_shadow: ProbeResponse,
    canaries: tuple[str, ...] = (),
) -> EscapeOracle:
    """Return ``escapes(present)`` for ``localize``.

    For a subset of present fragment ids it replays those fragments + the probe
    messages, reads the output structurally, and returns whether the residual
    against ``clean_shadow`` is non-empty (the subset exposes something the clean
    baseline did not).
    """
    by_id = {f.fragment_id: f for f in fragments}

    def escapes(present: frozenset[str]) -> bool:
        context = [by_id[fid].message for fid in by_id if fid in present]
        raw = run_inference(context + probe_messages)
        snapshot = structural_readout(raw, canaries)
        return residual_payloads(snapshot, clean_shadow).escape_detected

    return escapes
