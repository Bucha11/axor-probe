from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from axor_probe.shadow.context import CanonicalizedContextSummary


def _new_snapshot_id() -> str:
    # Opaque, non-reversible to raw snapshot content.
    return uuid.uuid4().hex


@dataclass(frozen=True)
class StateSnapshot:
    """
    Point-in-time, read-only capture of live main agent context.
    Ephemeral by default — never persisted.

    Explicitly absent — no handles passed to ProbeExecutor:
    - memory write handles
    - tool executor references
    - user output channel
    - session transcript writer

    Lifecycle: created → used for probe execution → discarded.
    Only redacted comparison outputs are persisted.
    """

    session_id: str
    timestamp: float
    context_window: tuple[dict[str, Any], ...]  # bounded selected context slice; Any because message values are provider-specific and untyped at this layer
    system_prompt_hash: str                       # hash only — never plain text
    canonicalized_summary: CanonicalizedContextSummary
    snapshot_id: str = field(default_factory=_new_snapshot_id)
