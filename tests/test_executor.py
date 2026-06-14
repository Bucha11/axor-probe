from __future__ import annotations

import logging
from typing import Any

from axor_probe.executor.runner import ProbeExecutor
from axor_probe.executor.snapshot import StateSnapshot
from axor_probe.probes.library import ProbeLibrary
from axor_probe.shadow.context import CanonicalizedContextSummary


def _summary(taint: bool) -> CanonicalizedContextSummary:
    return CanonicalizedContextSummary(
        trusted_user=False,
        prior_authorization=False,
        data_sensitivity="high" if taint else "low",
        escalation_chain="none",
        external_reads_count=1 if taint else 0,
        taint_active=taint,
        session_depth=1,
    )


def _snapshot(taint: bool, content: str = "raw secret content") -> StateSnapshot:
    return StateSnapshot(
        session_id="sess-1",
        timestamp=1000.0,
        context_window=({"role": "user", "content": content},),
        system_prompt_hash="abc",
        canonicalized_summary=_summary(taint),
    )


async def _capture_messages() -> tuple[ProbeExecutor, list]:
    seen: list = []

    async def inference_fn(messages: list[dict[str, Any]]) -> dict[str, Any]:
        seen.append(messages)
        return {"decision": "decline"}

    return ProbeExecutor(inference_fn), seen


async def test_warns_once_on_tainted_context_without_sanitizer(caplog) -> None:
    executor, _ = await _capture_messages()
    probe = ProbeLibrary().select()
    with caplog.at_level(logging.WARNING, logger="axor.probe.executor"):
        await executor.execute(probe, _snapshot(taint=True))
        await executor.execute(probe, _snapshot(taint=True))  # second time: no repeat
    warnings = [r for r in caplog.records if "TAINTED context" in r.message]
    assert len(warnings) == 1


async def test_no_warning_when_context_clean(caplog) -> None:
    executor, _ = await _capture_messages()
    probe = ProbeLibrary().select()
    with caplog.at_level(logging.WARNING, logger="axor.probe.executor"):
        await executor.execute(probe, _snapshot(taint=False))
    assert not [r for r in caplog.records if "TAINTED context" in r.message]


async def test_sanitizer_scrubs_replayed_context_and_suppresses_warning(caplog) -> None:
    seen: list = []

    async def inference_fn(messages: list[dict[str, Any]]) -> dict[str, Any]:
        seen.append(messages)
        return {"decision": "decline"}

    executor = ProbeExecutor(inference_fn, context_sanitizer=lambda s: "[SCRUBBED]")
    probe = ProbeLibrary().select()
    with caplog.at_level(logging.WARNING, logger="axor.probe.executor"):
        await executor.execute(probe, _snapshot(taint=True, content="raw secret content"))

    replayed = seen[0]
    assert replayed[0]["content"] == "[SCRUBBED]"
    assert "raw secret content" not in str(replayed)
    # With a sanitizer wired, no warning is emitted.
    assert not [r for r in caplog.records if "TAINTED context" in r.message]
