"""
Tests for the axor-core session-tap adapter (axor_probe/integration/core_tap.py).

P-34: axor-probe must not import axor-core. The core SessionContextView is
therefore stubbed locally as a small frozen dataclass (_FakeView) exposing the
same attributes — exactly how a structural Protocol attachment is exercised.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from axor_probe.executor.snapshot import StateSnapshot
from axor_probe.integration.core_tap import CoreContextTap, ViewSnapshotFactory
from axor_probe.pipeline.orchestrator import RuntimeEvent
from axor_probe.shadow.context import CanonicalizedContextSummary


# ── Local stub of core's SessionContextView (P-34: no core import) ─────────────

@dataclass(frozen=True)
class _FakeView:
    session_id: str = "sess-1"
    agent_id: str = "agent-1"
    timestamp: float = 1000.0
    turn_index: int = 3
    token_count: int = 42
    context_window: tuple[dict[str, object], ...] = (
        {"kind": "message", "source": "user", "content": "hi", "turn": 1},
    )
    system_prompt_hash: str = "abc123"
    taint_active: bool = False
    taint_sources: tuple[str, ...] = ()
    taint_scope: str = "none"
    taint_intent_age: int = 0
    external_read_count: int = 0
    taint_canaries: tuple[str, ...] = ("LIVE-CANARY-7c4d",)


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _AlwaysScheduler:
    def evaluate(self, event: RuntimeEvent) -> bool:
        return True


class _NeverScheduler:
    def evaluate(self, event: RuntimeEvent) -> bool:
        return False


@dataclass
class _RecordingPipeline:
    """Records calls to run() without doing any real work."""
    calls: list[RuntimeEvent] = field(default_factory=list)

    async def run(self, event: RuntimeEvent) -> None:
        self.calls.append(event)


@dataclass
class _SlowPipeline:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False

    async def run(self, event: RuntimeEvent) -> None:
        self.started.set()
        await self.release.wait()
        self.finished = True


# ── on_context_event: store + schedule ─────────────────────────────────────────

async def test_on_context_event_stores_latest_view() -> None:
    pipeline = _RecordingPipeline()
    tap = CoreContextTap(pipeline, _NeverScheduler())  # type: ignore[arg-type]
    view = _FakeView()

    await tap.on_context_event(view)  # type: ignore[arg-type]

    assert tap.latest_view is view


async def test_schedules_pipeline_run_when_scheduler_true() -> None:
    pipeline = _RecordingPipeline()
    tap = CoreContextTap(pipeline, _AlwaysScheduler())  # type: ignore[arg-type]

    await tap.on_context_event(_FakeView())  # type: ignore[arg-type]
    # Task was scheduled out-of-band; let the loop run it.
    await asyncio.sleep(0)

    assert len(pipeline.calls) == 1
    assert pipeline.calls[0].session_id == "sess-1"


async def test_does_not_schedule_when_scheduler_false() -> None:
    pipeline = _RecordingPipeline()
    tap = CoreContextTap(pipeline, _NeverScheduler())  # type: ignore[arg-type]

    await tap.on_context_event(_FakeView())  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert pipeline.calls == []
    assert tap.latest_view is not None  # view still stored


async def test_event_type_external_content_when_tainted() -> None:
    pipeline = _RecordingPipeline()
    tap = CoreContextTap(pipeline, _AlwaysScheduler())  # type: ignore[arg-type]

    await tap.on_context_event(_FakeView(taint_active=True))  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert pipeline.calls[0].event_type == "external_content"
    assert pipeline.calls[0].has_external_content is True


async def test_event_type_context_growth_when_clean() -> None:
    pipeline = _RecordingPipeline()
    tap = CoreContextTap(pipeline, _AlwaysScheduler())  # type: ignore[arg-type]

    await tap.on_context_event(_FakeView(taint_active=False))  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert pipeline.calls[0].event_type == "context_growth"


async def test_on_context_event_returns_promptly_without_awaiting_run() -> None:
    """on_context_event must not block on a slow pipeline.run."""
    pipeline = _SlowPipeline()
    tap = CoreContextTap(pipeline, _AlwaysScheduler())  # type: ignore[arg-type]

    await tap.on_context_event(_FakeView())  # type: ignore[arg-type]
    # The slow run is still in-flight (not finished) because we never awaited it.
    assert pipeline.finished is False

    # Confirm it really was scheduled and can complete out-of-band.
    await asyncio.wait_for(pipeline.started.wait(), timeout=1.0)
    pipeline.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pipeline.finished is True


# ── ViewSnapshotFactory.create ─────────────────────────────────────────────────

async def test_create_raises_when_no_view_captured() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)
    event = RuntimeEvent(session_id="s", agent_id="a", model="m", event_type="manual")

    try:
        await factory.create(event)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when no view captured")


async def test_create_maps_view_to_snapshot() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    view = _FakeView()
    await tap.on_context_event(view)  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)
    event = RuntimeEvent(session_id="sess-1", agent_id="agent-1", model="m", event_type="manual")

    snapshot = await factory.create(event)

    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.session_id == "sess-1"
    assert snapshot.timestamp == 1000.0
    assert snapshot.context_window == view.context_window
    assert snapshot.system_prompt_hash == "abc123"
    assert snapshot.canaries == ("LIVE-CANARY-7c4d",)  # live taint canaries carried through


async def test_summary_low_sensitivity_clean_session() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    await tap.on_context_event(_FakeView(taint_active=False, external_read_count=0, turn_index=3))  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)

    summary = (await factory.create(_make_event())).canonicalized_summary
    assert isinstance(summary, CanonicalizedContextSummary)
    assert summary.data_sensitivity == "low"
    assert summary.taint_active is False
    assert summary.external_reads_count == 0
    assert summary.session_depth == 3
    assert summary.trusted_user is False
    assert summary.prior_authorization is False
    assert summary.escalation_chain == "none"


async def test_summary_high_sensitivity_taint_and_reads() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    await tap.on_context_event(_FakeView(taint_active=True, external_read_count=2))  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)

    summary = (await factory.create(_make_event())).canonicalized_summary
    assert summary.data_sensitivity == "high"
    assert summary.taint_active is True
    assert summary.external_reads_count == 2


async def test_summary_medium_sensitivity_taint_no_reads() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    await tap.on_context_event(_FakeView(taint_active=True, external_read_count=0))  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)

    summary = (await factory.create(_make_event())).canonicalized_summary
    assert summary.data_sensitivity == "medium"


async def test_summary_medium_sensitivity_many_reads_no_taint() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    await tap.on_context_event(_FakeView(taint_active=False, external_read_count=3))  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)

    summary = (await factory.create(_make_event())).canonicalized_summary
    assert summary.data_sensitivity == "medium"


async def test_summary_session_depth_floor_is_one() -> None:
    tap = CoreContextTap(_RecordingPipeline(), _NeverScheduler())  # type: ignore[arg-type]
    await tap.on_context_event(_FakeView(turn_index=0))  # type: ignore[arg-type]
    factory = ViewSnapshotFactory(tap)

    summary = (await factory.create(_make_event())).canonicalized_summary
    assert summary.session_depth == 1


def _make_event() -> RuntimeEvent:
    return RuntimeEvent(session_id="sess-1", agent_id="agent-1", model="m", event_type="manual")
