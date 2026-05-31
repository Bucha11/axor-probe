from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from axor_probe.comparator.accumulator import DriftAccumulator
from axor_probe.comparator.scorer import ComparatorConfig, ComparisonMode, drift_score, should_triangulate
from axor_probe.comparator.semantic import SemanticJudge, SemanticJudgeResult
from axor_probe.comparator.structural import compare_payloads, ComparisonResult
from axor_probe.comparator.triangulator import triangulate, TriangulatedResult
from axor_probe.executor.runner import ProbeExecutor, ProbeResponse
from axor_probe.executor.snapshot import StateSnapshot
from axor_probe.probes.library import ProbeLibrary
from axor_probe.probes.schema import Probe, ProbeType
from axor_probe.probes.validator import ProbeValidator, StructuralAnomalyType
from axor_probe.shadow.context import CanonicalizedContextSummary
from axor_probe.shadow.instance import ContextlessShadowBaseline, ShadowInstance
from axor_probe.shadow.isolation import ShadowIsolation, comparator_confidence
from axor_probe.signals.drift import DriftAction, DriftSignal
from axor_probe.signals.report import ProbeReport
from axor_probe.storage.audit import ProbeAuditStore
from axor_probe.storage.redactor import PayloadRedactor

log = logging.getLogger("axor.probe.pipeline")


# ── Runtime event ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RuntimeEvent:
    """Triggering event from the live agent runtime."""
    session_id: str
    agent_id: str
    model: str
    event_type: str  # "context_growth" | "external_content" | "reputation" | "manual"
    token_count: int = 0
    has_external_content: bool = False


# ── Dependency protocols ──────────────────────────────────────────────────────

class ProbeScheduler(Protocol):
    """Gate: returns True if this event warrants a probe dispatch."""
    def evaluate(self, event: RuntimeEvent) -> bool: ...


class SnapshotFactory(Protocol):
    """Creates a read-only StateSnapshot from the live session context."""
    async def create(self, event: RuntimeEvent) -> StateSnapshot: ...


class ShadowInstanceFactory(Protocol):
    """Wraps ShadowInstance + ContextlessShadowBaseline execution and isolation metadata."""
    async def execute(self, probe: Probe, summary: CanonicalizedContextSummary) -> ProbeResponse: ...
    async def execute_baseline(self, probe: Probe) -> ProbeResponse: ...
    def confidence(self) -> float: ...


class Comparator(Protocol):
    """Wraps structural comparison, semantic scoring, and triangulation."""

    def compare(
        self,
        snapshot: ProbeResponse,
        shadow: ProbeResponse,
        probe_id: str,
        probe_type: ProbeType,
        probe_library_version: str,
        structural_anomaly: StructuralAnomalyType | None,
    ) -> ComparisonResult: ...

    async def score(
        self,
        result: ComparisonResult,
        probe_type: ProbeType,
        summary: CanonicalizedContextSummary | None = None,
    ) -> tuple[float, SemanticJudgeResult]: ...

    def should_triangulate(self, score: float, probe_type: ProbeType) -> bool: ...

    def triangulate_decisions(
        self,
        snapshot_decision: str,
        shadow_decision: str,
        baseline_decision: str,
    ) -> TriangulatedResult: ...


class IntegrationLayer(Protocol):
    """Downstream integration: sentinel, eval, core."""
    async def emit(self, report: ProbeReport, action: DriftAction) -> None: ...


# ── Concrete default implementations ─────────────────────────────────────────

@dataclass
class DefaultShadowInstanceFactory:
    """Wraps ShadowInstance + ContextlessShadowBaseline with isolation metadata."""
    shadow: ShadowInstance
    baseline: ContextlessShadowBaseline
    isolation: ShadowIsolation = field(default_factory=ShadowIsolation)

    async def execute(self, probe: Probe, summary: CanonicalizedContextSummary) -> ProbeResponse:
        return await self.shadow.execute(probe, summary)

    async def execute_baseline(self, probe: Probe) -> ProbeResponse:
        return await self.baseline.execute(probe)

    def confidence(self) -> float:
        return comparator_confidence(self.isolation)


@dataclass
class DefaultComparator:
    """Wraps structural comparison + semantic judge into the Comparator interface.

    The semantic judge is an *inference call* and may be backed by an external
    model. Payloads are redacted before they are handed to it (P-16/P-18) — the
    structural divergence used for local scoring is computed on the raw payloads,
    but nothing un-redacted leaves the process boundary.
    """
    semantic_judge: SemanticJudge
    redactor: PayloadRedactor = field(default_factory=PayloadRedactor)
    config: ComparatorConfig = field(default_factory=ComparatorConfig)

    def compare(
        self,
        snapshot: ProbeResponse,
        shadow: ProbeResponse,
        probe_id: str,
        probe_type: ProbeType,
        probe_library_version: str,
        structural_anomaly: StructuralAnomalyType | None,
    ) -> ComparisonResult:
        return compare_payloads(
            snapshot, shadow, probe_id, probe_type, probe_library_version, structural_anomaly
        )

    async def score(
        self,
        result: ComparisonResult,
        probe_type: ProbeType,
        summary: CanonicalizedContextSummary | None = None,
    ) -> tuple[float, SemanticJudgeResult]:
        # Redact BEFORE the judge inference call — the judge must never see raw
        # reasoning / disclosed content (the order used to be reversed: judge at
        # step 7, redaction only at step 10 before storage).
        judge_result = await self.semantic_judge.judge(
            self.redactor.redact(result.snapshot_payload),
            self.redactor.redact(result.shadow_payload),
            summary,
        )
        return drift_score(result, probe_type, judge_result), judge_result

    def should_triangulate(self, score: float, probe_type: ProbeType) -> bool:
        return should_triangulate(score, probe_type, self.config)

    def triangulate_decisions(
        self,
        snapshot_decision: str,
        shadow_decision: str,
        baseline_decision: str,
    ) -> TriangulatedResult:
        return triangulate(snapshot_decision, shadow_decision, baseline_decision)


# ── Orchestrator ──────────────────────────────────────────────────────────────

@dataclass
class ProbePipeline:
    """
    Single entry point for the probe cycle (O-1).
    All mutable state lives in injected components; ProbePipeline itself is stateless
    except for session-scoped counters needed for ProbeReport aggregation.
    """

    scheduler: ProbeScheduler
    snapshot_factory: SnapshotFactory
    executor: ProbeExecutor
    shadow_factory: ShadowInstanceFactory
    comparator: Comparator
    accumulator: DriftAccumulator
    redactor: PayloadRedactor
    audit_store: ProbeAuditStore
    library: ProbeLibrary = field(default_factory=ProbeLibrary)
    validator: ProbeValidator = field(default_factory=ProbeValidator)
    integrations: list[IntegrationLayer] = field(default_factory=list)
    calibration_status: str = "UNCALIBRATED"

    _probes_sent: int = field(default=0, init=False, repr=False)
    _probes_invalid: int = field(default=0, init=False, repr=False)
    _probes_triangulated: int = field(default=0, init=False, repr=False)
    _summary_calibration_anomalies: int = field(default=0, init=False, repr=False)
    _drift_signals: list[DriftSignal] = field(default_factory=list, init=False, repr=False)
    _timeline: list[ComparisonResult] = field(default_factory=list, init=False, repr=False)
    _active_session_id: str | None = field(default=None, init=False, repr=False)

    async def run(self, event: RuntimeEvent) -> ProbeReport | None:  # noqa: PLR0912
        """
        Execute one probe cycle for the given RuntimeEvent (O-1).
        Returns ProbeReport on success, None if the cycle was skipped or failed early.
        """

        # Step 1: gate
        if not self.scheduler.evaluate(event):
            return None
        self._ensure_session_scope(event.session_id)

        # Step 2: capture context
        try:
            snapshot = await self.snapshot_factory.create(event)
        except Exception:
            log.error("SNAPSHOT_FAILURE session=%s", event.session_id, exc_info=True)
            return None

        # Step 3: select probe
        try:
            probe = self.library.select()
        except Exception:
            log.error("PROBE_SELECTION_FAILURE session=%s", event.session_id, exc_info=True)
            return None

        # Step 4: validate probe
        pre_result = self.validator.pre_dispatch(probe)
        if not pre_result.valid:
            log.warning("PROBE_INVALID probe_id=%s detail=%s", probe.probe_id, pre_result.detail)
            self._probes_invalid += 1
            return None

        # Step 5: parallel dispatch — any failure discards the probe (O-4)
        # Save identifiers before discarding snapshot (O-5).
        snapshot_id = snapshot.snapshot_id
        summary = snapshot.canonicalized_summary
        try:
            snapshot_resp, shadow_resp = await asyncio.gather(
                self.executor.execute(probe, snapshot),
                self.shadow_factory.execute(probe, summary),
            )
        except Exception:
            log.error("EXECUTOR_FAILURE or SHADOW_FAILURE probe_id=%s", probe.probe_id, exc_info=True)
            return None
        finally:
            del snapshot  # O-5: StateSnapshot discarded immediately after step 5

        # Step 6: validate responses — structural anomaly degrades score, does not abort
        snap_val = self.validator.post_response(probe, snapshot_resp.raw)
        shad_val = self.validator.post_response(probe, shadow_resp.raw)
        structural_anomaly: StructuralAnomalyType | None = snap_val.anomaly or shad_val.anomaly

        # Step 7: compare — BINARY
        comparison = self.comparator.compare(
            snapshot_resp, shadow_resp,
            probe.probe_id, probe.probe_type, probe.probe_library_version,
            structural_anomaly,
        )
        comparison.session_id = event.session_id

        # Score computed once; reused for triangulation gate and final signal.
        score, judge_result = await self.comparator.score(comparison, probe.probe_type, summary)

        # Step 7b: triangulate if score falls in ambiguity band
        triangulation_result: TriangulatedResult | None = None
        comparison_mode = ComparisonMode.BINARY
        shadow_baseline_payload: dict[str, Any] | None = None

        if self.comparator.should_triangulate(score, probe.probe_type):
            try:
                baseline_resp = await self.shadow_factory.execute_baseline(probe)
                triangulation_result = self.comparator.triangulate_decisions(
                    snapshot_resp.decision,
                    shadow_resp.decision,
                    baseline_resp.decision,
                )
                comparison_mode = ComparisonMode.TRIANGULATED
                shadow_baseline_payload = {
                    "decision": baseline_resp.decision,
                    "reasoning": baseline_resp.reasoning,
                    "disclosed_data": baseline_resp.disclosed_data,
                    "policy_cited": baseline_resp.policy_cited,
                    "escalation_suggested": baseline_resp.escalation_suggested,
                    "tool_call_attempted": baseline_resp.tool_call_attempted,
                }
                self._probes_triangulated += 1
                if triangulation_result.classification.value == "summary_calibration_anomaly":
                    self._summary_calibration_anomalies += 1
            except Exception:
                log.warning("TRIANGULATION_SKIPPED probe_id=%s", probe.probe_id, exc_info=True)

        # Step 8: score + accumulate
        comparison.drift_score = score
        self.accumulator.record(comparison)
        longitudinal = self.accumulator.longitudinal_signal()
        consistency_anomaly = self.accumulator.check_consistency_anomaly()
        self._probes_sent += 1

        # Step 9: build unredacted signal
        recommended_action = DriftAction.from_longitudinal_signal(longitudinal, self.calibration_status)

        raw_signal = DriftSignal(
            signal_id=uuid.uuid4().hex,
            probe_id=probe.probe_id,
            probe_library_version=probe.probe_library_version,
            snapshot_id=snapshot_id,
            session_id=event.session_id,
            agent_id=event.agent_id,
            probe_type=probe.probe_type,
            divergence_category=comparison.divergence_category,
            drift_score=score,
            comparator_confidence=self.shadow_factory.confidence(),
            comparison_mode=comparison_mode,
            triangulation_result=triangulation_result,
            longitudinal_signal=longitudinal,
            field_divergences=tuple(comparison.field_divergences),
            semantic_judge_result=judge_result,
            snapshot_payload=comparison.snapshot_payload,
            shadow_payload=comparison.shadow_payload,
            shadow_baseline_payload=shadow_baseline_payload,
            calibration_status=self.calibration_status,
            timestamp=time.time(),
            recommended_action=recommended_action,
        )

        # Step 10: redact — hard stop on failure; unredacted payload never persisted (O-2, O-3)
        try:
            signal = dataclasses.replace(
                raw_signal,
                snapshot_payload=self.redactor.redact(raw_signal.snapshot_payload),
                shadow_payload=self.redactor.redact(raw_signal.shadow_payload),
                shadow_baseline_payload=(
                    self.redactor.redact(raw_signal.shadow_baseline_payload)
                    if raw_signal.shadow_baseline_payload is not None
                    else None
                ),
                field_divergences=self.redactor.redact_divergences(raw_signal.field_divergences),
            )
        except Exception:
            log.critical(
                "REDACTION_FAILURE probe_id=%s session=%s — aborting; unredacted payload not persisted",
                probe.probe_id, event.session_id,
                exc_info=True,
            )
            return None  # O-3: only hard stop after step 5
        finally:
            del raw_signal  # unredacted signal must not survive this scope

        # Propagate redacted payloads to comparison for P-12 compliance before timeline append.
        comparison.snapshot_payload = signal.snapshot_payload
        comparison.shadow_payload = signal.shadow_payload
        self._timeline.append(comparison)
        self._drift_signals.append(signal)

        # Step 11: persist — non-fatal; report still returned on failure (O-7)
        try:
            self.audit_store.record(signal)
        except Exception:
            log.error(
                "STORAGE_FAILURE probe_id=%s session=%s",
                probe.probe_id, event.session_id,
                exc_info=True,
            )

        # Step 12: build report
        report = ProbeReport.build(
            session_id=event.session_id,
            agent_id=event.agent_id,
            model=event.model,
            probe_library_version=probe.probe_library_version,
            drift_signals=list(self._drift_signals),
            timeline=list(self._timeline),
            probes_sent=self._probes_sent,
            probes_invalid=self._probes_invalid,
            probes_triangulated=self._probes_triangulated,
            summary_calibration_anomalies=self._summary_calibration_anomalies,
            consistency_anomaly_detected=consistency_anomaly,
            calibration_status=self.calibration_status,
            longitudinal_signal=longitudinal,
        )

        # Step 13: integrations — parallel, partial failure does not block others (O-6)
        if self.integrations:
            results = await asyncio.gather(
                *[integration.emit(report, recommended_action) for integration in self.integrations],
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log.error("INTEGRATION_FAILURE integration[%d]: %s", i, result, exc_info=result)

        return report

    def _ensure_session_scope(self, session_id: str) -> None:
        if self._active_session_id == session_id:
            return

        self._active_session_id = session_id
        self._probes_sent = 0
        self._probes_invalid = 0
        self._probes_triangulated = 0
        self._summary_calibration_anomalies = 0
        self._drift_signals.clear()
        self._timeline.clear()
        self.accumulator = DriftAccumulator(
            session_id=session_id,
            probe_library_version=self.accumulator.probe_library_version,
        )
