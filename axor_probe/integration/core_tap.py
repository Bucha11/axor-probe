"""
Adapter for axor-core's neutral, read-only "session tap" seam.

axor-core defines a ContextTap Protocol (axor_core/contracts/observation.py) that
pushes a SessionContextView on each governance-path context event. This module
attaches axor-probe to that seam WITHOUT importing axor-core (invariant P-34): we
implement ContextTap structurally and only reference core types under
TYPE_CHECKING. This mirrors the existing structural-Protocol pattern in
axor_probe/integration/core.py (CoreDriftSink).

Two pieces:

  * CoreContextTap — structurally implements core's ContextTap. It stores the
    latest SessionContextView and, on each event, decides via the probe
    ProbeScheduler whether to dispatch a probe cycle. Crucially, the probe cycle
    is scheduled OUT-OF-BAND via asyncio.create_task and is NEVER awaited inline:
    core's contract requires on_context_event to return promptly and not block
    the governance path. We hold references to spawned tasks so they are not
    garbage-collected mid-flight.

  * ViewSnapshotFactory — structurally implements probe's SnapshotFactory. It
    reads the latest SessionContextView captured by a CoreContextTap and maps it
    into a probe StateSnapshot, deriving the CanonicalizedContextSummary
    structural buckets from the raw observation facts ("the consumer does the
    bucketing"). If no view has been captured yet it raises; the orchestrator
    treats snapshot_factory.create exceptions as SNAPSHOT_FAILURE and skips the
    cycle, which is the desired conservative behavior.
"""
from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from axor_probe.executor.snapshot import StateSnapshot
from axor_probe.pipeline.orchestrator import ProbePipeline, ProbeScheduler, RuntimeEvent
from axor_probe.shadow.context import CanonicalizedContextSummary

# axor-core's real type is axor_core.contracts.observation.SessionContextView; we
# never import it (P-34) and type against the structural Protocol below instead,
# so static analysis needs no axor-core install.


@runtime_checkable
class _SessionContextViewLike(Protocol):
    """Structural shape of core's SessionContextView (P-34: no core import).

    Only the attributes axor-probe consumes are declared here.
    """

    session_id: str
    agent_id: str
    timestamp: float
    turn_index: int
    token_count: int
    context_window: tuple[dict[str, object], ...]
    system_prompt_hash: str
    taint_active: bool
    external_read_count: int


class CoreContextTap:
    """Structural implementation of core's ContextTap (P-34: no core import).

    Stores the latest pushed view and dispatches probe cycles out-of-band.

    The scheduling invariant is load-bearing: on_context_event must return
    promptly. We evaluate the cheap, synchronous ProbeScheduler gate inline and,
    only if it passes, fire-and-forget the (potentially slow) ProbePipeline.run
    via asyncio.create_task. The task reference is retained until completion so
    the event loop does not drop it.
    """

    def __init__(self, pipeline: ProbePipeline, scheduler: ProbeScheduler) -> None:
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._latest_view: _SessionContextViewLike | None = None
        # Hold strong refs to in-flight tasks so they are not GC'd (see
        # asyncio.create_task docs — the loop only keeps a weak reference).
        self._tasks: set[asyncio.Task[object]] = set()

    @property
    def latest_view(self) -> _SessionContextViewLike | None:
        """Most recently observed SessionContextView, or None if none yet."""
        return self._latest_view

    async def on_context_event(self, view: _SessionContextViewLike) -> None:
        """Receive a context event from core. Returns promptly (never blocks).

        Stores the view for the SnapshotFactory, then schedules a probe cycle
        out-of-band when the scheduler gate passes.
        """
        self._latest_view = view

        event = self._build_event(view)
        # Cheap synchronous gate inline; expensive cycle off the governance path.
        if self._scheduler.evaluate(event):
            task = asyncio.create_task(self._pipeline.run(event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _build_event(view: _SessionContextViewLike) -> RuntimeEvent:
        # taint_active means external/untrusted content is live in context.
        event_type = "external_content" if view.taint_active else "context_growth"
        return RuntimeEvent(
            session_id=view.session_id,
            agent_id=view.agent_id,
            model="",  # core view carries no model identity; left empty.
            event_type=event_type,
            token_count=view.token_count,
            has_external_content=view.taint_active,
        )


class ViewSnapshotFactory:
    """Structural implementation of probe's SnapshotFactory.

    Maps the CoreContextTap's latest SessionContextView into a StateSnapshot,
    bucketing the raw observation facts into a CanonicalizedContextSummary.
    """

    def __init__(self, tap: CoreContextTap) -> None:
        self._tap = tap

    async def create(self, event: RuntimeEvent) -> StateSnapshot:
        view = self._tap.latest_view
        if view is None:
            # No observation captured yet — orchestrator logs SNAPSHOT_FAILURE
            # and skips the cycle (conservative, intended behavior).
            raise RuntimeError("no SessionContextView captured yet")

        return StateSnapshot(
            session_id=view.session_id,
            timestamp=view.timestamp,
            context_window=tuple(view.context_window),  # pass-through
            system_prompt_hash=view.system_prompt_hash,
            canonicalized_summary=self._summarize(view),
        )

    @staticmethod
    def _summarize(view: _SessionContextViewLike) -> CanonicalizedContextSummary:
        external = view.external_read_count
        taint = view.taint_active

        # Bucket data_sensitivity from raw facts.
        if taint and external > 0:
            sensitivity = "high"
        elif taint or external > 2:
            sensitivity = "medium"
        else:
            sensitivity = "low"

        return CanonicalizedContextSummary(
            # Conservative defaults: mid-session has no escalation/auth signal.
            trusted_user=False,
            prior_authorization=False,
            data_sensitivity=sensitivity,
            escalation_chain="none",
            external_reads_count=external,
            taint_active=taint,
            session_depth=max(1, view.turn_index),
        )
