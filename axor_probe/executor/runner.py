from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from axor_probe.probes.schema import InjectionMode, Probe
from axor_probe.executor.snapshot import StateSnapshot

log = logging.getLogger("axor.probe.executor")

# Inference callable injected by the caller — no hard SDK dependency.
# Receives a list of message dicts and returns a dict response.
# Contract: InferenceFn must detect tool_call blocks in the provider response
# and return tool_call_attempted=True when present; raw tool calls must not be forwarded.
InferenceFn = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ProbeResponse:
    decision: str
    reasoning: str
    disclosed_data: bool
    policy_cited: bool
    escalation_suggested: bool
    tool_call_attempted: bool  # serialised as intent only — never dispatched
    raw: dict[str, Any]       # Any: provider response shape is untyped at this layer


class ProbeExecutor:
    """
    Executes probe against a StateSnapshot. Constructs an ephemeral inference call.
    Side-effects are structurally impossible — no handles to memory, tools,
    or user output are present in the execution context.

    Tool calls emitted by the model are serialised as intent only (tool_call_attempted=True)
    and never dispatched.
    """

    def __init__(
        self,
        inference_fn: InferenceFn,
        context_sanitizer: Callable[[str], str] | None = None,
    ) -> None:
        self._inference_fn = inference_fn
        # Optional scrubber applied to replayed context content. SHOULD be set
        # when inference_fn is backed by an external/untrusted model — the probe
        # instance replays the live session's (raw) context window. With a
        # local/trusted model it may be left None.
        self._context_sanitizer = context_sanitizer
        self._warned_unsanitized_taint = False

    async def execute(self, probe: Probe, snapshot: StateSnapshot) -> ProbeResponse:
        # Replaying a tainted (untrusted-content-bearing) context window to the
        # inference model with no sanitizer means raw external content reaches that
        # model. Surface it once — the operator must wire a context_sanitizer when the
        # model is external. See the sanitizer note in __init__ and README.
        if (
            self._context_sanitizer is None
            and snapshot.canonicalized_summary.taint_active
            and not self._warned_unsanitized_taint
        ):
            log.warning(
                "replaying TAINTED context to probe inference with no context_sanitizer "
                "(session=%s) — raw untrusted content reaches the model; wire a "
                "context_sanitizer for external models",
                snapshot.session_id,
            )
            self._warned_unsanitized_taint = True

        messages = self._build_messages(probe, snapshot)
        raw = await self._inference_fn(messages)

        return ProbeResponse(
            decision=_string_or_empty(raw.get("decision")),
            reasoning=_string_or_empty(raw.get("reasoning")),
            disclosed_data=_bool_or_false(raw.get("disclosed_data")),
            policy_cited=_bool_or_false(raw.get("policy_cited")),
            escalation_suggested=_bool_or_false(raw.get("escalation_suggested")),
            tool_call_attempted=_bool_or_false(raw.get("tool_call_attempted")),
            raw=raw,
        )

    def _build_messages(
        self, probe: Probe, snapshot: StateSnapshot
    ) -> list[dict[str, Any]]:
        # Replay the bounded context window from the snapshot, scrubbing message
        # content first when a sanitizer is configured (external-model safety).
        messages: list[dict[str, Any]] = [
            self._sanitize_message(m) for m in snapshot.context_window
        ]

        if probe.inject_as == InjectionMode.USER_MESSAGE:
            messages.append({"role": "user", "content": probe.scenario})
        else:  # TOOL_RESULT
            # tool_use_id is a sentinel; InferenceFn implementations must adapt
            # this to the provider-specific message schema before sending.
            messages.append({"role": "tool", "tool_use_id": "probe_injection", "content": probe.scenario})

        return messages

    def _sanitize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._context_sanitizer is None:
            return message
        content = message.get("content")
        if not isinstance(content, str):
            return message
        return {**message, "content": self._context_sanitizer(content)}


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _bool_or_false(value: Any) -> bool:
    return value if isinstance(value, bool) else False
