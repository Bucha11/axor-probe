from __future__ import annotations

import dataclasses
import re
from typing import Any

from axor_probe.comparator.structural import FieldDivergence


# Fields stored in plain form — structural, categorical, non-sensitive
PLAIN_FIELDS = frozenset({
    "decision",
    "disclosed_data",
    "policy_cited",
    "escalation_suggested",
    "tool_call_attempted",
})

# Fields requiring scrubbing before storage
REDACTED_FIELDS = frozenset({
    "reasoning",
})

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


class PayloadRedactor:
    """
    Applied to all DriftSignal payloads before persistence.
    Redaction is best-effort — regex-based scrubbing; unknown sensitive content may survive.
    High-sensitivity deployments should store only categorical fields or encrypted payloads.
    """

    def redact(self, payload: dict[str, Any]) -> dict[str, Any]:  # Any: payload field values vary
        result: dict[str, Any] = {}
        for key, value in payload.items():
            result[key] = self.redact_field(key, value)
        return result

    def redact_field(self, key: str, value: Any) -> Any:  # Any: payload field values vary
        if key in PLAIN_FIELDS:
            return value
        if key in REDACTED_FIELDS:
            return self._scrub(str(value))
        return "[REDACTED:UNKNOWN_FIELD]"

    def redact_divergences(self, divergences: tuple[FieldDivergence, ...]) -> tuple[FieldDivergence, ...]:
        return tuple(
            dataclasses.replace(
                divergence,
                snapshot_value=self.redact_field(divergence.field_name, divergence.snapshot_value),
                shadow_value=self.redact_field(divergence.field_name, divergence.shadow_value),
            )
            for divergence in divergences
        )

    def _scrub(self, text: str) -> str:
        """Best-effort PII scrubbing. Not a compliance guarantee."""
        text = _EMAIL_RE.sub("[REDACTED:EMAIL]", text)
        text = _PHONE_RE.sub("[REDACTED:PHONE]", text)
        return text
