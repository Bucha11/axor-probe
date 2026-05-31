from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("axor.probe.controller")

# Dispatch callback: receives a probe_type hint (str) and fires a probe
DispatchFn = Callable[[str], Awaitable[None]]


@dataclass
class ProbeScheduleConfig:
    context_growth_tokens: int = 2000
    external_content_threshold: int = 500
    reputation_snapshot_path: str | None = None
    reputation_score_threshold: float = 0.5
    reputation_poll_interval_seconds: int = 60
    max_interval_seconds: int = 600
    max_probes_per_session: int = 10
    jitter_seconds: float = 15.0
    cooldown_window_seconds: int = 30  # coalesces concurrent triggers into one dispatch


@dataclass
class ProbeController:
    """
    Event-driven probe scheduler.

    Trigger coalescing: multiple triggers within cooldown_window_seconds produce
    exactly one dispatch (P-20).

    Jitter is mandatory — fixed-interval probing is prohibited (P-19).

    axor-probe never subscribes to events from other Axor components (P-26).
    """

    config: ProbeScheduleConfig
    dispatch_fn: DispatchFn
    _probes_sent: int = field(default=0, init=False, repr=False)
    _last_trigger_time: float = field(default=0.0, init=False, repr=False)
    _pending_dispatch: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def trigger(self, reason: str = "manual") -> None:
        """
        Signal that a probe should be dispatched.
        Coalesces duplicate triggers within cooldown_window_seconds.
        """
        async with self._lock:
            if self._probes_sent >= self.config.max_probes_per_session:
                log.debug("max_probes_per_session reached; trigger ignored")
                return

            now = time.monotonic()
            since_last = now - self._last_trigger_time
            if since_last < self.config.cooldown_window_seconds and self._pending_dispatch:
                log.debug("trigger coalesced (within cooldown window)")
                return

            self._pending_dispatch = True
            self._last_trigger_time = now

        jitter = random.uniform(0, self.config.jitter_seconds)
        await asyncio.sleep(jitter)

        async with self._lock:
            if not self._pending_dispatch:
                return  # superseded by a later trigger that reset the flag

            if self._probes_sent >= self.config.max_probes_per_session:
                self._pending_dispatch = False
                return

            self._pending_dispatch = False
            self._probes_sent += 1

        log.info("dispatching probe (reason=%s, total=%d)", reason, self._probes_sent)
        await self.dispatch_fn(reason)

    def evaluate(self, event: Any) -> bool:
        """
        Synchronous gate used by ProbePipeline.
        Accepts RuntimeEvent-like objects without importing pipeline.orchestrator.
        """
        if self._probes_sent >= self.config.max_probes_per_session:
            return False

        now = time.monotonic()
        since_last = now - self._last_trigger_time
        interval_due = self._last_trigger_time == 0.0 or since_last >= self.config.max_interval_seconds
        event_type = getattr(event, "event_type", "")
        token_count = int(getattr(event, "token_count", 0))
        has_external_content = bool(getattr(event, "has_external_content", False))

        should_dispatch = (
            event_type == "manual"
            or token_count >= self.config.context_growth_tokens
            or (has_external_content and token_count >= self.config.external_content_threshold)
            or interval_due
        )
        if not should_dispatch or since_last < self.config.cooldown_window_seconds:
            return False

        self._last_trigger_time = now
        self._probes_sent += 1
        return True

    async def run_reputation_poll(self) -> None:
        """
        Polls reputation snapshot file if configured (P-25).
        Absent snapshot silently disables the trigger.
        axor-probe never subscribes to external component events (P-26).
        """
        if self.config.reputation_snapshot_path is None:
            return

        while True:
            await asyncio.sleep(self.config.reputation_poll_interval_seconds)
            path = self.config.reputation_snapshot_path
            if not os.path.exists(path):
                continue
            try:
                data = await asyncio.to_thread(lambda: json.loads(pathlib.Path(path).read_text()))
                score = float(data.get("reputation_score", 1.0))
                if score < self.config.reputation_score_threshold:
                    await self.trigger("reputation_below_threshold")
            except Exception:
                log.debug("reputation snapshot read failed; skipping trigger")

    @property
    def probes_sent(self) -> int:
        return self._probes_sent
