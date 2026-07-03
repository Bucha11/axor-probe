from __future__ import annotations

import json
from typing import Any

from axor_probe.executor.readout import structural_readout
from axor_probe.executor.runner import InferenceFn, ProbeResponse, canary_fragment
from axor_probe.probes.schema import Probe
from axor_probe.shadow.context import CanonicalizedContextSummary


def _summary_to_text(summary: CanonicalizedContextSummary) -> str:
    return json.dumps({
        "trusted_user": summary.trusted_user,
        "prior_authorization": summary.prior_authorization,
        "data_sensitivity": summary.data_sensitivity,
        "escalation_chain": summary.escalation_chain,
        "external_reads_count": summary.external_reads_count,
        "taint_active": summary.taint_active,
        "session_depth": summary.session_depth,
    })


class ShadowInstance:
    """
    Isolated instance receiving: system prompt + CanonicalizedContextSummary + probe.
    No raw session content, no tool state, no memory.
    Initialised fresh per probe call — never persisted between probes (P-15).

    The canary is planted here too (same confidential record as the snapshot side),
    so the clean shadow *could* leak it — the differential is whether the tainted
    context makes the snapshot leak it while the clean shadow does not.
    """

    def __init__(self, inference_fn: InferenceFn, system_prompt: str) -> None:
        self._inference_fn = inference_fn
        self._system_prompt = system_prompt

    async def execute(
        self, probe: Probe, summary: CanonicalizedContextSummary
    ) -> ProbeResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Context summary: {_summary_to_text(summary)}\n\nScenario: {probe.scenario}"},
        ]
        if probe.canary:
            messages.append(canary_fragment(probe.canary))
        raw = await self._inference_fn(messages)
        return structural_readout(raw, probe.canary)


class ContextlessShadowBaseline:
    """
    Isolated instance receiving: system prompt + probe only.
    No CanonicalizedContextSummary, no context, no canary planted (P-14) — being
    contextless, it cannot hold or leak the secret. Used only during triangulation.
    Initialised fresh per probe call — never persisted (P-15).
    """

    def __init__(self, inference_fn: InferenceFn, system_prompt: str) -> None:
        self._inference_fn = inference_fn
        self._system_prompt = system_prompt

    async def execute(self, probe: Probe) -> ProbeResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": probe.scenario},
        ]
        raw = await self._inference_fn(messages)
        return structural_readout(raw, probe.canary)
