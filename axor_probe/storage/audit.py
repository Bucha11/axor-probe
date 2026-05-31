from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_probe.signals.drift import DriftSignal


@dataclass
class ProbeAuditStore:
    """
    In-memory audit store for DriftSignals.
    Separate from session transcript store by design — no session state reference.
    All stored signals must have passed through PayloadRedactor before record() is called.
    Production deployments should inject a persistent backend.
    """

    _signals: list[DriftSignal] = field(default_factory=list, repr=False)

    def record(self, signal: DriftSignal) -> None:
        """Store a redacted DriftSignal. Caller is responsible for redacting payloads first."""
        self._signals.append(signal)

    def all_signals(self) -> list[DriftSignal]:
        """Return a snapshot of all recorded signals."""
        return list(self._signals)

    def signals_for_session(self, session_id: str) -> list[DriftSignal]:
        return [s for s in self._signals if s.session_id == session_id]
