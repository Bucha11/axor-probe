from __future__ import annotations

import json
from typing import Any

from axor_probe.executor.runner import InferenceFn, ProbeResponse
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


def _extract_response(raw: dict[str, Any]) -> ProbeResponse:  # Any: provider response shape is untyped at this layer
    return ProbeResponse(
        decision=_string_or_empty(raw.get("decision")),
        reasoning=_string_or_empty(raw.get("reasoning")),
        disclosed_data=_bool_or_false(raw.get("disclosed_data")),
        policy_cited=_bool_or_false(raw.get("policy_cited")),
        escalation_suggested=_bool_or_false(raw.get("escalation_suggested")),
        tool_call_attempted=_bool_or_false(raw.get("tool_call_attempted")),
        raw=raw,
    )


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _bool_or_false(value: Any) -> bool:
    return value if isinstance(value, bool) else False


class ShadowInstance:
    """
    Isolated instance receiving: system prompt + CanonicalizedContextSummary + probe.
    No raw session content, no tool state, no memory.
    Initialised fresh per probe call — never persisted between probes (P-15).
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
        raw = await self._inference_fn(messages)
        return _extract_response(raw)


class ContextlessShadowBaseline:
    """
    Isolated instance receiving: system prompt + probe only.
    No CanonicalizedContextSummary. No context (P-14).
    Used only during triangulated comparison.
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
        return _extract_response(raw)
