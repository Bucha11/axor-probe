from __future__ import annotations

import asyncio
from types import SimpleNamespace

from axor_probe.controller.scheduler import ProbeController, ProbeScheduleConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_controller(
    max_probes: int = 10,
    cooldown: int = 30,
    jitter: float = 0.0,  # zero jitter for deterministic tests
) -> tuple[ProbeController, list[str]]:
    dispatched: list[str] = []

    async def dispatch(reason: str) -> None:
        dispatched.append(reason)

    cfg = ProbeScheduleConfig(
        max_probes_per_session=max_probes,
        cooldown_window_seconds=cooldown,
        jitter_seconds=jitter,
    )
    return ProbeController(cfg, dispatch), dispatched


# ── Trigger coalescing (P-20) ─────────────────────────────────────────────────

async def test_concurrent_triggers_coalesced_to_one() -> None:
    ctrl, dispatched = _make_controller(cooldown=30)
    await asyncio.gather(ctrl.trigger("t1"), ctrl.trigger("t2"), ctrl.trigger("t3"))
    assert len(dispatched) == 1


async def test_triggers_after_cooldown_window_dispatch_separately() -> None:
    ctrl, dispatched = _make_controller(cooldown=0)  # zero cooldown = no coalescing window
    await ctrl.trigger("first")
    await ctrl.trigger("second")
    assert len(dispatched) == 2


# ── max_probes_per_session hard cap ───────────────────────────────────────────

async def test_max_probes_cap_enforced() -> None:
    ctrl, dispatched = _make_controller(max_probes=2, cooldown=0)
    for i in range(5):
        ctrl._last_trigger_time = 0.0
        await ctrl.trigger(f"t{i}")
    assert len(dispatched) == 2


async def test_probes_sent_property_tracks_count() -> None:
    ctrl, _ = _make_controller(max_probes=3, cooldown=0)
    for i in range(3):
        ctrl._last_trigger_time = 0.0
        await ctrl.trigger(f"t{i}")
    assert ctrl.probes_sent == 3


async def test_trigger_ignored_at_cap() -> None:
    ctrl, dispatched = _make_controller(max_probes=1, cooldown=0)
    await ctrl.trigger("first")
    await ctrl.trigger("second")  # should be ignored
    assert len(dispatched) == 1


# ── Jitter (P-19) ────────────────────────────────────────────────────────────

async def test_dispatch_occurs_after_jitter_sleep() -> None:
    # With non-zero jitter, dispatch still happens (just delayed)
    dispatched: list[str] = []

    async def dispatch(reason: str) -> None:
        dispatched.append(reason)

    cfg = ProbeScheduleConfig(
        max_probes_per_session=1,
        cooldown_window_seconds=0,
        jitter_seconds=0.01,  # tiny but non-zero
    )
    ctrl = ProbeController(cfg, dispatch)
    await ctrl.trigger("jitter_test")
    assert len(dispatched) == 1


async def test_two_sequential_dispatches_can_have_different_delays() -> None:
    """
    Jitter is random — two sequential dispatches should not always take the same time.
    We verify the code calls random.uniform (structural test, not probabilistic).
    """
    import random
    delays: list[float] = []
    _orig = random.uniform

    def capture_uniform(a: float, b: float) -> float:
        result = _orig(a, b)
        delays.append(result)
        return result

    ctrl, _ = _make_controller(max_probes=5, cooldown=0, jitter=1.0)

    random.uniform = capture_uniform  # type: ignore[assignment]
    try:
        for _ in range(3):
            ctrl._last_trigger_time = 0.0
            await ctrl.trigger("jitter")
    finally:
        random.uniform = _orig  # type: ignore[assignment]

    assert len(delays) == 3, "random.uniform called once per dispatch"
    # Not all delays need to be different (probability), but the function was called


# ── Reputation poll disabled when path=None (P-25) ───────────────────────────

async def test_reputation_poll_disabled_when_no_path() -> None:
    ctrl, dispatched = _make_controller()
    # run_reputation_poll should return immediately with no path
    await asyncio.wait_for(ctrl.run_reputation_poll(), timeout=0.1)
    assert len(dispatched) == 0


# ── No session termination (P-27) ────────────────────────────────────────────

async def test_trigger_does_not_terminate_session() -> None:
    ctrl, dispatched = _make_controller(cooldown=0)
    # Just verify it runs without raising and doesn't touch any session attribute
    await ctrl.trigger("safe")
    assert not hasattr(ctrl, "session_terminated")
    assert not hasattr(ctrl, "session_id")


def test_evaluate_manual_event_dispatches_once() -> None:
    ctrl, _ = _make_controller(cooldown=0)
    event = SimpleNamespace(event_type="manual", token_count=0, has_external_content=False)
    assert ctrl.evaluate(event)
    assert ctrl.probes_sent == 1


def test_evaluate_context_growth_threshold() -> None:
    ctrl, _ = _make_controller(cooldown=0)
    event = SimpleNamespace(event_type="context_growth", token_count=2000, has_external_content=False)
    assert ctrl.evaluate(event)


def test_evaluate_external_content_threshold() -> None:
    ctrl, _ = _make_controller(cooldown=0)
    event = SimpleNamespace(event_type="external_content", token_count=500, has_external_content=True)
    assert ctrl.evaluate(event)
