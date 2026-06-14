# axor-probe

Runtime behavioral drift detection via shadow instance comparison.

`axor-probe` detects context-induced reasoning drift in LLM agents while a
session is running. It captures a read-only snapshot of the live session,
executes behavioral probes out-of-band against a snapshot-derived probe
instance and an isolated shadow instance, compares their structured JSON
responses, and emits redacted drift telemetry.

The central question is:

> Same probe, same policy, operationally equivalent context: does the snapshot
> probe instance respond consistently with an isolated shadow baseline?

This is probabilistic behavioral telemetry, not formal verification. It does
not prove compromise, intent, or absolute safety. It measures relative
behavioral consistency under policy pressure.

## What It Is For

`axor-probe` is designed to observe whether accumulated context changes an
agent's behavioral disposition in ways that differ from a clean relative
baseline. It is useful for:

- detecting policy-pressure divergence during long-running sessions
- surfacing possible prompt-injection or context-taint effects
- feeding behavioral integrity signals into audit, sentinel, or core
  enforcement layers
- tracking drift longitudinally across multiple probes in a session

It is not a task evaluator. It does not measure whether the agent completed the
user task correctly. It asks whether the agent responds consistently when the
same policy-pressure probe is applied to comparable isolated contexts.

## Non-Goals

`axor-probe` does not:

- inject probes into the live main agent
- write to session memory
- execute tools
- produce user-visible output
- write probe responses into the session transcript
- detect weight-level backdoors when both instances behave identically
- detect consistently unsafe behavior when both instances are equally unsafe
- replace offline calibration

The live session is never touched by probe execution.

## Architecture

```text
Live Main Agent
  |
  | read-only snapshot at trigger time
  v
StateSnapshot
  |
  | probe + canonicalized context summary
  +--------------------------+
  |                          |
  v                          v
ProbeExecutor           ShadowInstance
snapshot probe          isolated instance
instance                with canonical summary
  |                          |
  +------------+-------------+
               |
               v
          Comparator
          BINARY by default
          TRIANGULATED on ambiguity
               |
               v
          DriftSignal
               |
               v
          PayloadRedactor
               |
               v
          ProbeAuditStore / integrations
```

The main tradeoff is intentional:

- Side-effect isolation is gained. Probes cannot affect memory, tools,
  transcript, or user output.
- Exact live-session fidelity is lost. A snapshot-derived probe instance is
  operationally similar to the live agent, but not identical.

Results are therefore evidence of drift tendency in a controlled out-of-band
environment, not proof of live-session behavior.

## Execution Model

The runtime path is:

```text
StateSnapshot -> ProbeExecutor -> ProbeResponse -> Comparator
              -> DriftSignal -> PayloadRedactor -> ProbeAuditStore
```

`StateSnapshot` is ephemeral and read-only. It contains:

- `session_id`
- `timestamp`
- bounded `context_window`
- `system_prompt_hash`, never the plain system prompt
- `CanonicalizedContextSummary`
- opaque `snapshot_id`

It explicitly does not contain:

- memory write handles
- tool executor references
- user output channels
- transcript writers

`InjectionMode` controls framing inside the ephemeral inference call only:

- `USER_MESSAGE` frames the probe as a user turn
- `TOOL_RESULT` frames the probe as a tool result

Neither mode delivers anything to the live session.

## Probe Responses

Probe responses are structured JSON. The default probe library expects:

```json
{
  "decision": "decline",
  "reasoning": "policy forbids disclosure",
  "disclosed_data": false,
  "policy_cited": true,
  "escalation_suggested": false,
  "tool_call_attempted": false
}
```

`tool_call_attempted` is intent-only. If a model emits a tool call during probe
execution, the injected inference adapter must serialize that fact into this
field and must not dispatch the tool call.

`ProbeValidator` runs in two stages:

- `pre_dispatch`: validates the probe itself. Invalid probes are not executed.
- `post_response`: validates response shape and JSON types. Structural failures
  become comparator anomalies rather than immediate session actions.

## Context Summary

The shadow instance does not receive raw live-session content. It receives a
structural `CanonicalizedContextSummary`:

```python
CanonicalizedContextSummary(
    trusted_user=True,
    prior_authorization=False,
    data_sensitivity="low",
    escalation_chain="none",
    external_reads_count=0,
    taint_active=False,
    session_depth=1,
)
```

This summary contains shape only, never raw strings from the live session.

This creates unavoidable context asymmetry: the snapshot probe instance has a
bounded context window, while the shadow has only structural context. Legitimate
contextual reasoning can therefore look like drift. Lazy triangulation exists to
reduce this false-positive class.

## Comparison Modes

### BINARY

Default mode. Compares:

```text
snapshot probe instance vs shadow instance with CanonicalizedContextSummary
```

The structural comparator deterministically checks configured fields such as
`decision`, `disclosed_data`, `policy_cited`, and `reasoning`.

### TRIANGULATED

Triggered only when the binary drift score falls within the ambiguity band of
the probe-type threshold. It adds one more inference call:

```text
snapshot probe instance
vs shadow instance with CanonicalizedContextSummary
vs contextless shadow baseline
```

Typical interpretation:

- snapshot and shadow agree, baseline differs: context explains behavior
- snapshot differs from shadow and baseline: likely drift signal
- all three agree: no diagnostic signal
- shadow differs from snapshot and baseline: summary calibration anomaly

Triangulation is lazy because it costs an extra inference call.

## Scoring

Scoring combines:

- deterministic field divergence
- semantic judge output
- probe-type threshold
- longitudinal session accumulation

When a borderline result is triangulated, the triangulation verdict is folded back
into the action-driving score (`effective_drift_score`): a `LEGITIMATE` result (context
explains the divergence) is damped and a `SUMMARY_CALIBRATION_ANOMALY` is dropped to
zero (P-31), while `DRIFT_SIGNAL`/`NO_SIGNAL` keep the binary score. Triangulation can
only lower the score, never raise it — its purpose is false-positive reduction. The raw
binary comparison remains available on the `DriftSignal.triangulation_result` for audit.

All numeric weights and thresholds are marked `UNCALIBRATED`. Current values are
first-principles estimates and must be fitted against labeled data before
production enforcement.

`DriftAction` is determined by longitudinal signal, not by a single probe:

- `LOG_ONLY`
- `ELEVATED_REVIEW`
- `RESTRICTED_MODE`

`RESTRICTED_MODE` is only allowed when `calibration_status == "CALIBRATED"`.
When uncalibrated, strong signals are downgraded to `ELEVATED_REVIEW`.

`axor-probe` never initiates session termination. Session termination, if any,
belongs to `axor-core` degradation policy.

## Redaction and Storage

Only redacted comparison outputs should reach storage.

Plain structural fields:

- `decision`
- `disclosed_data`
- `policy_cited`
- `escalation_suggested`
- `tool_call_attempted`

Scrubbed fields:

- `reasoning`

Unknown fields are redacted by default as `[REDACTED:UNKNOWN_FIELD]`.

`PayloadRedactor` is best-effort. It includes basic email and phone scrubbing,
but it is not a compliance guarantee. High-sensitivity deployments should store
only categorical fields, hashes, or encrypted payloads behind audit-gated
access.

`ProbeAuditStore` is separate from the session transcript store.

## Probe Library

The built-in library is versioned as `1.0.0` and includes at least 23 probes:

| Probe type | Minimum variants | Focus |
| --- | ---: | --- |
| `DATA_DISCLOSURE` | 5 | claimed authority, data sensitivity, requestor framing |
| `SCOPE_EXPANSION` | 4 | adjacent task expansion, urgency, implicit permissions |
| `AUTHORITY_ESCALATION` | 4 | user and tool-result escalation framing |
| `BUDGET_BYPASS` | 3 | budget pressure and sub-agent delegation |
| `INSTRUCTION_CONFLICT` | 4 | policy conflict and override attempts |
| `IDENTITY_PROBE` | 3 | role consistency and constraint acknowledgement |

Cross-version probe results are excluded from longitudinal accumulation.

## Scheduling

`ProbeController` supports:

- context-growth triggers
- external-content triggers
- manual triggers
- optional reputation snapshot polling
- jitter before dispatch
- cooldown-window coalescing
- `max_probes_per_session` hard cap

Fixed-interval probing is intentionally avoided. Multiple triggers inside
`cooldown_window_seconds` coalesce into one dispatch.

`axor-probe` does not subscribe directly to other Axor component events.
Callers pass runtime events or wire dispatch callbacks.

## Integrations

The package has no required inference SDK dependency and no hard dependency on
`axor-core`, `axor-sentinel`, or `axor-eval`. Integration modules use protocols
and injected callbacks.

Dependency direction is one-way:

```text
axor-probe -> axor-core
axor-probe -> axor-sentinel
axor-probe -> axor-eval
axor-core  -X-> axor-probe
axor-sentinel -X-> axor-probe
```

Available integration helpers:

- `integration.core.notify_core(signal, sink)`
- `integration.sentinel.emit_to_sentinel(signal, sink)`
- `integration.eval.feed_audit(report, feed_fn)`

Core and sentinel integrations emit only for significant actions
(`ELEVATED_REVIEW` or calibrated `RESTRICTED_MODE`). `LOG_ONLY` remains local
telemetry.

## Minimal Wiring Sketch

```python
from axor_probe.comparator.accumulator import DriftAccumulator
from axor_probe.comparator.semantic import SemanticJudge
from axor_probe.executor.runner import ProbeExecutor
from axor_probe.pipeline.orchestrator import (
    DefaultComparator,
    DefaultShadowInstanceFactory,
    ProbePipeline,
)
from axor_probe.shadow.instance import ContextlessShadowBaseline, ShadowInstance
from axor_probe.storage.audit import ProbeAuditStore
from axor_probe.storage.redactor import PayloadRedactor


async def inference_fn(messages):
    # Adapter around your model provider.
    # Must return the canonical probe response dict.
    return {
        "decision": "decline",
        "reasoning": "policy forbids this",
        "disclosed_data": False,
        "policy_cited": True,
        "escalation_suggested": False,
        "tool_call_attempted": False,
    }


pipeline = ProbePipeline(
    scheduler=my_scheduler,
    snapshot_factory=my_snapshot_factory,
    executor=ProbeExecutor(inference_fn),
    shadow_factory=DefaultShadowInstanceFactory(
        shadow=ShadowInstance(inference_fn, system_prompt=my_system_prompt),
        baseline=ContextlessShadowBaseline(inference_fn, system_prompt=my_system_prompt),
    ),
    comparator=DefaultComparator(SemanticJudge(inference_fn)),
    accumulator=DriftAccumulator(session_id="session-1", probe_library_version="1.0.0"),
    redactor=PayloadRedactor(),
    audit_store=ProbeAuditStore(),
)

report = await pipeline.run(runtime_event)
```

Production callers are expected to provide:

- snapshot factory for the live session runtime
- model-provider inference adapter
- persistent audit backend if in-memory storage is insufficient
- calibrated thresholds before automated enforcement

## Offline Calibration

Before production enforcement:

1. Collect labeled known-good and drift-injected sessions.
2. Run the probe library across both sets in BINARY mode.
3. Measure false-positive rate, true-positive rate, precision, and recall per
   probe type.
4. Choose an explicit false-positive target.
5. Calibrate field weights, thresholds, probe-type multipliers,
   `ambiguity_band`, and longitudinal coefficients.
6. Calibrate `CONSISTENCY_ANOMALY_THRESHOLD` against expected variance at the
   operational sampling settings.
7. Mark the resulting configuration as `CALIBRATED` with dataset version and
   date.

Until calibration, `RESTRICTED_MODE` must not trigger automatically.

## Known Limitations

- Out-of-band fidelity gap: snapshot execution is not identical to live-agent
  execution.
- Context asymmetry: structural summaries cannot encode all legitimate context.
- Inference cost: at least two calls per probe, plus a third on triangulation.
- Probe recognizability: static probes can become learnable over time.
- Redaction is best-effort.
- Relative baseline only: shared model-level compromise may be invisible.
- Consistent unsafe behavior may look consistent rather than divergent.
- Comparator difficulty remains real; high-stakes signals require review.

## Package Structure

```text
axor_probe/
  probes/        Probe schemas, validator, and versioned library
  executor/      StateSnapshot and out-of-band ProbeExecutor
  shadow/        Shadow instances, context summary, isolation confidence
  controller/    Probe scheduling and trigger coalescing
  comparator/    Structural comparison, semantic judge, scoring, triangulation
  storage/       Redaction and audit storage
  signals/       DriftSignal and ProbeReport models
  calibration/   Offline calibration pipeline stub
  integration/   Optional core, sentinel, and eval adapters
```

## Development

```bash
python -m pytest tests -q
ruff check axor_probe tests
```

The base package intentionally has zero required runtime dependencies. Dev
dependencies are listed under `project.optional-dependencies.dev`.
