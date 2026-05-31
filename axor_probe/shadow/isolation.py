from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowIsolation:
    shared_memory: bool = False
    shared_tool_state: bool = False
    shared_raw_context: bool = False
    same_model_version: bool = True
    same_sampling_config: bool = True       # sampling mismatch invalidates comparison
    same_system_prompt_hash: bool = True
    same_tool_schema_hash: bool = True      # schema affects reasoning even without execution
    same_output_schema: bool = True
    receives_canonical_summary: bool = True


def comparator_confidence(isolation: ShadowIsolation) -> float:
    """
    Degrades comparator confidence based on isolation violations.
    Sampling config mismatch (* 0.5) invalidates the comparison.
    System prompt hash mismatch (* 0.3) is a near-fatal violation — shadow policy differs.
    All multipliers are UNCALIBRATED.
    """
    base = 1.0
    if isolation.shared_memory:
        base *= 0.3
    if isolation.shared_tool_state:
        base *= 0.6
    if isolation.shared_raw_context:
        base *= 0.5
    if not isolation.same_sampling_config:
        base *= 0.5
    if not isolation.same_model_version:
        base *= 0.4
    if not isolation.same_system_prompt_hash:
        base *= 0.3
    if not isolation.same_tool_schema_hash:
        base *= 0.7
    if not isolation.same_output_schema:
        base *= 0.8
    if not isolation.receives_canonical_summary:
        base *= 0.8
    return base
