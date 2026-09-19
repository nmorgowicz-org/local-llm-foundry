# Draft: future Sixcat evaluation integration

Status: high-level parking-lot plan. This is not an implementation request
for Local LLM Foundry yet.

## Intent

Evaluate model changes and LoRA merges with the upstream
[`vcruz305/sixcat-eval`](https://github.com/vcruz305/sixcat-eval) suite. The
Foundry integration should point Sixcat at a selected OpenAI-compatible model
endpoint and retain useful result receipts without coupling the evaluator to
Foundry internals.

## Proposed boundary

- Sixcat remains an upstream dependency whose source and revision are kept
  separate from Foundry-specific integration code.
- Local LLM Foundry owns optional endpoint orchestration and product
  presentation.
- Results retain the Sixcat revision, model identity, endpoint settings,
  sampling policy, and raw receipts.

## Candidate integration shape

1. Pin a reviewed Sixcat revision and document a clean update workflow.
2. Define a Foundry manifest for endpoint, model ID, sampling policy, task
   limit, timeout, and receipt destination.
3. Add an optional adapter for starting or discovering an OpenAI-compatible
   local endpoint.
4. Keep evaluation external to the UI/runtime process, with explicit timeout,
   cancellation, and redaction behavior.
5. Display or link retained receipts rather than flattening protocol-specific
   metrics into one score.

## Questions to resolve later

- Should Foundry launch Sixcat locally, delegate to a worker, or only ingest
  completed receipts?
- What minimum cross-platform contract is required for Windows GPU serving and
  macOS evaluation orchestration?
- Which Sixcat result fields and receipt format are stable enough to publish
  as a Foundry integration API?

## Gates before implementation

- A reviewed Sixcat invocation contract exists.
- Endpoint identity and model identity can be verified rather than inferred
  from filenames or aliases.
- Failed, partial, resumed, and unsupported measurements are represented
  explicitly.
- Security review covers API keys, local endpoint exposure, subprocesses, and
  generated artifacts.
- A small end-to-end fixture proves Foundry can consume a retained receipt
  without importing evaluator implementation modules.
