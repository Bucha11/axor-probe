from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProbeType(str, Enum):
    DATA_DISCLOSURE = "data_disclosure"
    SCOPE_EXPANSION = "scope_expansion"
    AUTHORITY_ESCALATION = "authority_escalation"
    BUDGET_BYPASS = "budget_bypass"
    INSTRUCTION_CONFLICT = "instruction_conflict"
    IDENTITY_PROBE = "identity_probe"


class InjectionMode(str, Enum):
    USER_MESSAGE = "user_message"
    TOOL_RESULT = "tool_result"
    # SYSTEM_NOTE excluded — risks conflating policy layer with probe content


@dataclass(frozen=True)
class PolicyPressure:
    # Validation of constraint and stress_type (e.g. rejecting empty strings)
    # is intentionally delegated to ProbeValidator.pre_dispatch, not enforced here.
    constraint: str
    stress_type: str


@dataclass(frozen=True)
class Probe:
    probe_id: str
    probe_type: ProbeType
    probe_library_version: str
    scenario: str
    policy_pressure: PolicyPressure
    expected_schema: dict[str, str]  # values are JSON type descriptors: 'string', 'boolean', 'number', etc.
    inject_as: InjectionMode
    timestamp: float
    # Planted high-entropy secret for disclosure probes; the readout flags
    # disclosed_data structurally when it appears in the output. None → no canary
    # (the probe's structural signal is the tool-call attempt instead).
    canary: str | None = None
