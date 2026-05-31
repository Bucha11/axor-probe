from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalizedContextSummary:
    # Structural signals only — no raw strings from the live session.
    # Validation of value semantics is delegated to ProbeController.
    trusted_user: bool
    prior_authorization: bool
    data_sensitivity: str      # "low" | "medium" | "high" | "critical"
    escalation_chain: str      # "none" | "pending" | "active"
    external_reads_count: int
    taint_active: bool
    session_depth: int
