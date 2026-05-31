from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock


from axor_probe.comparator.accumulator import DriftAccumulator
from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.comparator.semantic import SemanticJudgeResult
from axor_probe.comparator.structural import compare_payloads, ComparisonResult
from axor_probe.comparator.triangulator import DriftClassification, TriangulatedResult
from axor_probe.executor.runner import ProbeExecutor, ProbeResponse
from axor_probe.executor.snapshot import StateSnapshot
from axor_probe.pipeline.orchestrator import (
    ProbePipeline,
    RuntimeEvent,
)
from axor_probe.probes.library import ProbeLibrary
from axor_probe.shadow.context import CanonicalizedContextSummary
from axor_probe.signals.drift import DriftAction, DriftSignal
from axor_probe.signals.report import ProbeReport, VERDICT_INCONCLUSIVE
from axor_probe.storage.audit import ProbeAuditStore
from axor_probe.storage.redactor import PayloadRedactor


# ── Shared fixtures ───────────────────────────────────────────────────────────

_SUMMARY = CanonicalizedContextSummary(
    trusted_user=True,
    prior_authorization=False,
    data_sensitivity="low",
    escalation_chain="none",
    external_reads_count=0,
    taint_active=False,
    session_depth=1,
)

_SNAPSHOT = StateSnapshot(
    session_id="sess-1",
    timestamp=1000.0,
    context_window=({"role": "user", "content": "hello"},),
    system_prompt_hash="abc123",
    canonicalized_summary=_SUMMARY,
)

_CONSISTENT_RESPONSE: dict[str, Any] = {
    "decision": "decline",
    "reasoning": "policy forbids this",
    "disclosed_data": False,
    "policy_cited": True,
    "escalation_suggested": False,
    "tool_call_attempted": False,
}


def _make_probe_response(**overrides: Any) -> ProbeResponse:
    base = dict(
        decision="decline",
        reasoning="policy forbids this",
        disclosed_data=False,
        policy_cited=True,
        escalation_suggested=False,
        tool_call_attempted=False,
        raw=dict(_CONSISTENT_RESPONSE),
    )
    base.update(overrides)
    return ProbeResponse(**base)


class _AlwaysProbeScheduler:
    def evaluate(self, event: RuntimeEvent) -> bool:
        return True


class _NeverProbeScheduler:
    def evaluate(self, event: RuntimeEvent) -> bool:
        return False


class _SnapshotFactory:
    async def create(self, event: RuntimeEvent) -> StateSnapshot:
        return _SNAPSHOT


class _FailingSnapshotFactory:
    async def create(self, event: RuntimeEvent) -> StateSnapshot:
        raise RuntimeError("snapshot failure")


class _FakeShadowFactory:
    def __init__(
        self,
        response: ProbeResponse | None = None,
        baseline: ProbeResponse | None = None,
        fail_main: bool = False,
        fail_baseline: bool = False,
    ) -> None:
        self._response = response or _make_probe_response()
        self._baseline = baseline or _make_probe_response()
        self._fail_main = fail_main
        self._fail_baseline = fail_baseline

    async def execute(self, probe: Any, summary: Any) -> ProbeResponse:
        if self._fail_main:
            raise RuntimeError("shadow failure")
        return self._response

    async def execute_baseline(self, probe: Any) -> ProbeResponse:
        if self._fail_baseline:
            raise RuntimeError("baseline failure")
        return self._baseline

    def confidence(self) -> float:
        return 1.0


class _FakeComparator:
    """Returns a clean consistent comparison — no divergences."""

    def compare(self, snapshot: Any, shadow: Any, probe_id: str, probe_type: Any,
                probe_library_version: str, structural_anomaly: Any) -> ComparisonResult:
        return compare_payloads(snapshot, shadow, probe_id, probe_type, probe_library_version, structural_anomaly)

    async def score(self, result: ComparisonResult, probe_type: Any, summary: Any = None) -> tuple[float, SemanticJudgeResult]:
        return 0.0, SemanticJudgeResult(
            policy_ref_match=True,
            decision_direction_match=True,
            context_contradiction=False,
        )

    def should_triangulate(self, score: float, probe_type: Any) -> bool:
        return False

    def triangulate_decisions(self, snap: str, shad: str, base: str) -> TriangulatedResult:
        return TriangulatedResult(
            snapshot_decision=snap,
            shadow_decision=shad,
            baseline_decision=base,
            classification=DriftClassification.NO_SIGNAL,
        )


class _HighDriftComparator(_FakeComparator):
    """Returns high drift score to trigger DRIFT_DETECTED verdict."""

    async def score(self, result: ComparisonResult, probe_type: Any, summary: Any = None) -> tuple[float, SemanticJudgeResult]:
        return 0.9, SemanticJudgeResult(
            policy_ref_match=False,
            decision_direction_match=False,
            context_contradiction=True,
        )


class _AmbiguousComparator(_FakeComparator):
    """Returns score in ambiguity band to trigger triangulation."""

    async def score(self, result: ComparisonResult, probe_type: Any, summary: Any = None) -> tuple[float, SemanticJudgeResult]:
        return 0.5, SemanticJudgeResult(True, True, False)

    def should_triangulate(self, score: float, probe_type: Any) -> bool:
        return True


def _make_pipeline(
    scheduler: Any = None,
    snapshot_factory: Any = None,
    shadow_factory: Any = None,
    comparator: Any = None,
    audit_store: Any = None,
    integrations: list = None,
) -> tuple[ProbePipeline, ProbeAuditStore]:
    store = audit_store or ProbeAuditStore()
    pipeline = ProbePipeline(
        scheduler=scheduler or _AlwaysProbeScheduler(),
        snapshot_factory=snapshot_factory or _SnapshotFactory(),
        executor=ProbeExecutor(AsyncMock(return_value=_CONSISTENT_RESPONSE)),
        shadow_factory=shadow_factory or _FakeShadowFactory(),
        comparator=comparator or _FakeComparator(),
        accumulator=DriftAccumulator("sess-1", "1.0.0"),
        redactor=PayloadRedactor(),
        audit_store=store,
        integrations=integrations or [],
    )
    return pipeline, store


_EVENT = RuntimeEvent(
    session_id="sess-1",
    agent_id="agent-1",
    model="test-model",
    event_type="manual",
)


# ── Gate ──────────────────────────────────────────────────────────────────────

async def test_gate_returns_none_when_scheduler_says_no() -> None:
    pipeline, store = _make_pipeline(scheduler=_NeverProbeScheduler())
    result = await pipeline.run(_EVENT)
    assert result is None
    assert len(store.all_signals()) == 0


# ── Snapshot failure ──────────────────────────────────────────────────────────

async def test_snapshot_failure_returns_none() -> None:
    pipeline, store = _make_pipeline(snapshot_factory=_FailingSnapshotFactory())
    result = await pipeline.run(_EVENT)
    assert result is None
    assert len(store.all_signals()) == 0


# ── Shadow failure (O-4) ──────────────────────────────────────────────────────

async def test_shadow_failure_returns_none_o4() -> None:
    pipeline, store = _make_pipeline(shadow_factory=_FakeShadowFactory(fail_main=True))
    result = await pipeline.run(_EVENT)
    assert result is None
    assert len(store.all_signals()) == 0


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_run_returns_probe_report_on_success() -> None:
    pipeline, store = _make_pipeline()
    report = await pipeline.run(_EVENT)

    assert report is not None
    assert isinstance(report, ProbeReport)
    assert report.session_id == "sess-1"
    assert report.probes_sent == 1
    assert report.probes_invalid == 0


async def test_signal_stored_in_audit_store() -> None:
    pipeline, store = _make_pipeline()
    await pipeline.run(_EVENT)

    signals = store.all_signals()
    assert len(signals) == 1
    assert signals[0].session_id == "sess-1"


# ── Payload redaction (O-2) ────────────────────────────────────────────────────

async def test_stored_signal_has_no_unredacted_reasoning() -> None:
    shadow_resp = _make_probe_response(reasoning="call user@example.com about salary")
    pipeline, store = _make_pipeline(
        shadow_factory=_FakeShadowFactory(response=shadow_resp),
    )
    await pipeline.run(_EVENT)

    signal = store.all_signals()[0]
    assert "user@example.com" not in signal.shadow_payload.get("reasoning", "")
    assert "[REDACTED:EMAIL]" in signal.shadow_payload.get("reasoning", "")


# ── Redaction failure → hard stop (O-3) ──────────────────────────────────────

async def test_redaction_failure_returns_none_and_does_not_persist_o3() -> None:
    class _BrokenRedactor(PayloadRedactor):
        def redact(self, payload: Any) -> Any:
            raise RuntimeError("redaction exploded")

    pipeline, store = _make_pipeline()
    pipeline.redactor = _BrokenRedactor()
    result = await pipeline.run(_EVENT)

    assert result is None
    assert len(store.all_signals()) == 0  # unredacted payload must not be persisted


# ── Storage failure — non-fatal (O-7) ─────────────────────────────────────────

async def test_storage_failure_still_returns_report_o7() -> None:
    class _BrokenStore(ProbeAuditStore):
        def record(self, signal: DriftSignal) -> None:
            raise RuntimeError("write failed")

    pipeline, _ = _make_pipeline(audit_store=_BrokenStore())
    report = await pipeline.run(_EVENT)

    assert report is not None
    assert report.probes_sent == 1


# ── Integrations (O-6) — partial failure does not block others ────────────────

async def test_integration_partial_failure_does_not_abort_o6() -> None:
    calls: list[str] = []

    class _FailIntegration:
        async def emit(self, report: ProbeReport, action: DriftAction) -> None:
            raise RuntimeError("integration down")

    class _OkIntegration:
        async def emit(self, report: ProbeReport, action: DriftAction) -> None:
            calls.append("ok")

    pipeline, _ = _make_pipeline(integrations=[_FailIntegration(), _OkIntegration()])
    report = await pipeline.run(_EVENT)

    assert report is not None
    assert "ok" in calls  # second integration still ran despite first failing


# ── DriftAction mapping ───────────────────────────────────────────────────────

async def test_high_drift_produces_elevated_or_restricted_action() -> None:
    pipeline, _ = _make_pipeline(comparator=_HighDriftComparator())
    # Run multiple probes to push longitudinal signal above 0.3
    for _ in range(5):
        report = await pipeline.run(_EVENT)

    assert report is not None
    assert report.overall_verdict in ("DRIFT_DETECTED", "CONSISTENT", "INCONCLUSIVE")
    # Recommended action on final signal should escalate past LOG_ONLY
    last_signal = pipeline._drift_signals[-1]
    assert last_signal.recommended_action in (DriftAction.ELEVATED_REVIEW, DriftAction.RESTRICTED_MODE)


# ── Triangulation (TRIANGULATED mode) ────────────────────────────────────────

async def test_triangulation_runs_when_score_in_ambiguity_band() -> None:
    pipeline, store = _make_pipeline(comparator=_AmbiguousComparator())
    report = await pipeline.run(_EVENT)

    assert report is not None
    assert report.probes_triangulated == 1
    signal = store.all_signals()[0]
    assert signal.comparison_mode == ComparisonMode.TRIANGULATED


async def test_triangulation_skip_on_baseline_failure_uses_binary() -> None:
    pipeline, store = _make_pipeline(
        comparator=_AmbiguousComparator(),
        shadow_factory=_FakeShadowFactory(fail_baseline=True),
    )
    report = await pipeline.run(_EVENT)

    assert report is not None
    assert report.probes_triangulated == 0
    signal = store.all_signals()[0]
    assert signal.comparison_mode == ComparisonMode.BINARY


# ── Probe counter ─────────────────────────────────────────────────────────────

async def test_probes_sent_increments_across_runs() -> None:
    pipeline, _ = _make_pipeline()
    await pipeline.run(_EVENT)
    await pipeline.run(_EVENT)
    assert pipeline._probes_sent == 2


async def test_pipeline_state_resets_between_sessions() -> None:
    pipeline, _ = _make_pipeline()
    first = await pipeline.run(_EVENT)
    second = await pipeline.run(RuntimeEvent(
        session_id="sess-2",
        agent_id=_EVENT.agent_id,
        model=_EVENT.model,
        event_type=_EVENT.event_type,
    ))

    assert first is not None
    assert second is not None
    assert second.session_id == "sess-2"
    assert second.probes_sent == 1
    assert len(second.drift_signals) == 1
    assert second.drift_signals[0].session_id == "sess-2"


# ── PROBE_INVALID counter ─────────────────────────────────────────────────────

async def test_invalid_probe_increments_counter_and_returns_none() -> None:
    """ProbeLibrary that returns an invalid probe (empty probe_id)."""
    import dataclasses as dc
    from axor_probe.probes.schema import Probe

    class _BadLibrary(ProbeLibrary):
        def select(self) -> Probe:
            base = super().select()
            return dc.replace(base, probe_id="")  # pre_dispatch will reject this

    pipeline, store = _make_pipeline()
    pipeline.library = _BadLibrary()
    result = await pipeline.run(_EVENT)

    assert result is None
    assert pipeline._probes_invalid == 1
    assert len(store.all_signals()) == 0


# ── DriftAction.from_longitudinal_signal ─────────────────────────────────────

def test_drift_action_log_only_below_03() -> None:
    assert DriftAction.from_longitudinal_signal(0.0) == DriftAction.LOG_ONLY
    assert DriftAction.from_longitudinal_signal(0.29) == DriftAction.LOG_ONLY


def test_drift_action_elevated_between_03_and_07() -> None:
    assert DriftAction.from_longitudinal_signal(0.3) == DriftAction.ELEVATED_REVIEW
    assert DriftAction.from_longitudinal_signal(0.5) == DriftAction.ELEVATED_REVIEW
    assert DriftAction.from_longitudinal_signal(0.69) == DriftAction.ELEVATED_REVIEW


def test_drift_action_restricted_above_07() -> None:
    assert DriftAction.from_longitudinal_signal(0.7) == DriftAction.ELEVATED_REVIEW
    assert DriftAction.from_longitudinal_signal(1.0) == DriftAction.ELEVATED_REVIEW
    assert DriftAction.from_longitudinal_signal(0.7, "CALIBRATED") == DriftAction.RESTRICTED_MODE
    assert DriftAction.from_longitudinal_signal(1.0, "CALIBRATED") == DriftAction.RESTRICTED_MODE


# ── ProbeReport.build ─────────────────────────────────────────────────────────

def test_report_build_inconclusive_below_3_probes() -> None:
    report = ProbeReport.build(
        session_id="s", agent_id="a", model="m",
        probe_library_version="1.0.0",
        drift_signals=[], timeline=[],
        probes_sent=2, probes_invalid=0, probes_triangulated=0,
        summary_calibration_anomalies=0,
        consistency_anomaly_detected=False,
        calibration_status="UNCALIBRATED",
        longitudinal_signal=0.0,
    )
    assert report.overall_verdict == VERDICT_INCONCLUSIVE
