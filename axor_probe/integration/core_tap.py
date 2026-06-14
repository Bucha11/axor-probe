"""
Adapter for axor-core's neutral, read-only "session tap" seam.

FORWARD INTEGRATION: axor-core does not yet emit a per-turn session view or
call an on_context_event tap from GovernedNode — there is no such observation
contract or wiring in core today (no SessionContextView / ContextTap in
axor_core.contracts). Following the same pattern as
axor_sentinel/integration/core_sink.py (CoreSessionSink), this module defines
the shape it consumes HERE, consumer-side, as the structural Protocol
_SessionContextViewLike below — it does NOT import a core module that may not
exist (invariant P-34). A host adapter (or a future GovernedNode observation
hook) satisfies the tap by duck-typing: constructing a view exposing those
attributes and calling on_context_event at each governance-path context event.
Until such a producer exists CoreContextTap has no caller and is exercised only
by tests.

This mirrors the structural-Protocol pattern in axor_probe/integration/core.py
(CoreDriftSink → core's BehavioralDriftObserver, which core DOES implement) and
the forward-integration pattern in axor_sentinel/integration/core_sink.py
(CoreSessionSink, where core is likewise not yet a producer).

Two pieces:

  * CoreContextTap — the probe-side tap a core producer pushes views into. It
    stores the latest view and, on each event, decides via the probe
    ProbeScheduler whether to dispatch a probe cycle. Crucially, the probe cycle
    is scheduled OUT-OF-BAND via asyncio.create_task and is NEVER awaited inline:
    the tap contract requires on_context_event to return promptly and not block
    the governance path. We hold references to spawned tasks so they are not
    garbage-collected mid-flight.

  * ViewSnapshotFactory — structurally implements probe's SnapshotFactory. It
    reads the latest session view captured by a CoreContextTap and maps it
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

# There is no core type to import yet (see module docstring). We type against the
# structural Protocol below, which a future core-side observation view — or a host
# adapter standing in for one — satisfies by duck-typing. No axor-core install is
# required for static analysis (P-34).


@runtime_checkable
class _SessionContextViewLike(Protocol):
    """Structural shape of the per-turn session view this tap consumes.

    Defined consumer-side (P-34: no core import). Only the attributes axor-probe
    actually reads are declared here; a producer may expose a superset.
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
    """The probe-side context tap a core producer would push views into.

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
