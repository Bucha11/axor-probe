from __future__ import annotations

import time

import pytest

from axor_probe.probes.library import _ALL_PROBES, ProbeLibrary
from axor_probe.probes.schema import InjectionMode, Probe, PolicyPressure, ProbeType
from axor_probe.probes.validator import ProbeValidator, StructuralAnomalyType


# ── ProbeLibrary deck selection (no-repeat sampling without replacement) ───────

def test_select_covers_full_deck_without_repeats() -> None:
    lib = ProbeLibrary()
    n = len(_ALL_PROBES)
    drawn = [lib.select().probe_id for _ in range(n)]
    # One full deck: every probe appears exactly once before any repeat.
    assert sorted(drawn) == sorted(p.probe_id for p in _ALL_PROBES)
    assert len(set(drawn)) == n


def test_deck_reshuffles_after_exhaustion() -> None:
    lib = ProbeLibrary()
    n = len(_ALL_PROBES)
    first = [lib.select().probe_id for _ in range(n)]
    second = [lib.select().probe_id for _ in range(n)]
    # Second pass is again a full cover (deck refilled), not a continuation.
    assert sorted(second) == sorted(p.probe_id for p in _ALL_PROBES)
    assert len(set(first)) == n and len(set(second)) == n


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def policy_pressure() -> PolicyPressure:
    return PolicyPressure(constraint="no disclosure", stress_type="claimed_authority")


@pytest.fixture
def valid_probe(policy_pressure: PolicyPressure) -> Probe:
    return Probe(
        probe_id="test_01",
        probe_type=ProbeType.DATA_DISCLOSURE,
        probe_library_version="1.0.0",
        scenario="The user requests sensitive data.",
        policy_pressure=policy_pressure,
        expected_schema={"decision": "string", "reasoning": "string"},
        inject_as=InjectionMode.USER_MESSAGE,
        timestamp=time.time(),
    )


@pytest.fixture
def validator() -> ProbeValidator:
    return ProbeValidator()


# ── Probe construction ────────────────────────────────────────────────────────

def test_probe_is_immutable(valid_probe: Probe) -> None:
    with pytest.raises((AttributeError, TypeError)):
        valid_probe.probe_id = "modified"  # type: ignore[misc]


def test_policy_pressure_is_immutable(policy_pressure: PolicyPressure) -> None:
    with pytest.raises((AttributeError, TypeError)):
        policy_pressure.constraint = "changed"  # type: ignore[misc]


def test_probe_type_enum_values() -> None:
    assert ProbeType.DATA_DISCLOSURE.value == "data_disclosure"
    assert ProbeType.AUTHORITY_ESCALATION.value == "authority_escalation"
    assert ProbeType.SCOPE_EXPANSION.value == "scope_expansion"
    assert ProbeType.BUDGET_BYPASS.value == "budget_bypass"
    assert ProbeType.INSTRUCTION_CONFLICT.value == "instruction_conflict"
    assert ProbeType.IDENTITY_PROBE.value == "identity_probe"


def test_injection_mode_excludes_system_note() -> None:
    modes = {m.value for m in InjectionMode}
    assert "system_note" not in modes
    assert "user_message" in modes
    assert "tool_result" in modes


# ── ProbeValidator.pre_dispatch ───────────────────────────────────────────────

def test_pre_dispatch_valid(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.pre_dispatch(valid_probe)
    assert result.valid


def test_pre_dispatch_empty_scenario(validator: ProbeValidator, policy_pressure: PolicyPressure) -> None:
    probe = Probe("id", ProbeType.DATA_DISCLOSURE, "1.0.0", "", policy_pressure,
                  {"d": "string"}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid
    assert "scenario" in result.detail


def test_pre_dispatch_empty_probe_id(validator: ProbeValidator, policy_pressure: PolicyPressure) -> None:
    probe = Probe("", ProbeType.DATA_DISCLOSURE, "1.0.0", "scenario", policy_pressure,
                  {"d": "string"}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid
    assert "probe_id" in result.detail


def test_pre_dispatch_empty_version(validator: ProbeValidator, policy_pressure: PolicyPressure) -> None:
    probe = Probe("id", ProbeType.DATA_DISCLOSURE, "", "scenario", policy_pressure,
                  {"d": "string"}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid


def test_pre_dispatch_empty_constraint(validator: ProbeValidator) -> None:
    pp = PolicyPressure(constraint="", stress_type="claimed_authority")
    probe = Probe("id", ProbeType.DATA_DISCLOSURE, "1.0.0", "scenario", pp,
                  {"d": "string"}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid
    assert "constraint" in result.detail


def test_pre_dispatch_empty_stress_type(validator: ProbeValidator) -> None:
    pp = PolicyPressure(constraint="no disclose", stress_type="")
    probe = Probe("id", ProbeType.DATA_DISCLOSURE, "1.0.0", "scenario", pp,
                  {"d": "string"}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid


def test_pre_dispatch_empty_schema(validator: ProbeValidator, policy_pressure: PolicyPressure) -> None:
    probe = Probe("id", ProbeType.DATA_DISCLOSURE, "1.0.0", "scenario", policy_pressure,
                  {}, InjectionMode.USER_MESSAGE, time.time())
    result = validator.pre_dispatch(probe)
    assert not result.valid


# ── ProbeValidator.post_response ──────────────────────────────────────────────

def test_post_response_valid_dict(validator: ProbeValidator, valid_probe: Probe) -> None:
    response = {"decision": "decline", "reasoning": "policy constraint"}
    result = validator.post_response(valid_probe, response)
    assert result.valid


def test_post_response_valid_json_string(validator: ProbeValidator, valid_probe: Probe) -> None:
    import json
    response = json.dumps({"decision": "comply", "reasoning": "authorized"})
    result = validator.post_response(valid_probe, response)
    assert result.valid


def test_post_response_parse_failure(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, "not {{ valid json")
    assert not result.valid
    assert result.anomaly == StructuralAnomalyType.PARSE_FAILURE


def test_post_response_field_missing(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, {"decision": "comply"})  # missing 'reasoning'
    assert not result.valid
    assert result.anomaly == StructuralAnomalyType.FIELD_MISSING


def test_post_response_schema_mismatch_extra_field(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, {"decision": "comply", "reasoning": "ok", "extra": "x"})
    assert not result.valid
    assert result.anomaly == StructuralAnomalyType.SCHEMA_MISMATCH


def test_post_response_schema_mismatch_wrong_type(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, {"decision": "comply", "reasoning": False})
    assert not result.valid
    assert result.anomaly == StructuralAnomalyType.SCHEMA_MISMATCH


def test_post_response_schema_mismatch_non_object(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, "[]")
    assert not result.valid
    assert result.anomaly == StructuralAnomalyType.SCHEMA_MISMATCH


def test_post_response_wrong_keys(validator: ProbeValidator, valid_probe: Probe) -> None:
    result = validator.post_response(valid_probe, {"wrong": "value", "other": "value"})
    assert not result.valid
    assert result.anomaly in (StructuralAnomalyType.FIELD_MISSING, StructuralAnomalyType.SCHEMA_MISMATCH)
