"""H3: the semantic judge (an inference call) must receive only redacted payloads."""

from __future__ import annotations

import json

import pytest

from axor_probe.comparator.semantic import SemanticJudge
from axor_probe.comparator.structural import ComparisonResult
from axor_probe.executor.runner import ProbeExecutor
from axor_probe.pipeline.orchestrator import DefaultComparator
from axor_probe.probes.schema import ProbeType


def _comparison_with_pii() -> ComparisonResult:
    payload = {
        "decision": "comply",
        "reasoning": "sent the report to alice@example.com per request",
        "disclosed_data": True,
        "policy_cited": False,
        "escalation_suggested": False,
        "tool_call_attempted": False,
    }
    return ComparisonResult(
        probe_id="p1",
        probe_type=ProbeType.DATA_DISCLOSURE,
        probe_library_version="v1",
        field_divergences=[],
        divergence_category=None,
        snapshot_payload=dict(payload),
        shadow_payload=dict(payload),
    )


@pytest.mark.asyncio
async def test_judge_receives_redacted_payload():
    captured: dict = {}

    async def fake_inference(messages):
        captured["messages"] = messages
        return {"policy_ref_match": True, "decision_direction_match": True,
                "context_contradiction": False}

    comparator = DefaultComparator(semantic_judge=SemanticJudge(fake_inference))
    await comparator.score(_comparison_with_pii(), ProbeType.DATA_DISCLOSURE)

    blob = json.dumps(captured["messages"])
    assert "alice@example.com" not in blob       # raw PII must not reach the judge
    assert "[REDACTED:EMAIL]" in blob            # scrubbed form is what it sees


@pytest.mark.asyncio
async def test_context_sanitizer_scrubs_replayed_context():
    captured: dict = {}

    async def fake_inference(messages):
        captured["messages"] = messages
        return {"decision": "ok"}

    class _Snap:
        context_window = [{"role": "user", "content": "secret token sk-ABC123"}]

    from axor_probe.probes.schema import InjectionMode, Probe, PolicyPressure

    probe = Probe(
        probe_id="p", probe_type=ProbeType.DATA_DISCLOSURE,
        probe_library_version="v1", scenario="check",
        inject_as=InjectionMode.USER_MESSAGE,
        policy_pressure=PolicyPressure(constraint="c", stress_type="s"),
        expected_schema={"decision": "string"},
        timestamp=0.0,
    )
    executor = ProbeExecutor(
        fake_inference,
        context_sanitizer=lambda s: s.replace("sk-ABC123", "[REDACTED]"),
    )
    await executor.execute(probe, _Snap())
    blob = json.dumps(captured["messages"])
    assert "sk-ABC123" not in blob
    assert "[REDACTED]" in blob
