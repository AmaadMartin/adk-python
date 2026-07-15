## Task

### User intent with respect to ADK
Implement logic to filter out non-persisted state keys (those prefixed with `temp:`) from being written to session events across checkpoints in the ADK Python workflow.

### Feature Description
In `adk-python`, specifically inside `src/google/adk/workflow/_node_runner.py`, the `_flush_deltas` and `_flush_output_and_deltas` methods move pending state and artifact deltas from the runtime Context onto the Event structure for persistence. Non-persisted state keys (like intermediate processing flags) are prefixed with `temp:` and should be dropped during this synchronization step so they aren't saved to standard checkpoints. 

### Use Cases & Examples
- A node sets `ctx.state["temp:cache"] = {"foo": "bar"}`. This state flows through the context for the duration of the run but when `_flush_deltas` is called, the `temp:cache` key is excluded from the Event's `state_delta` and thus is not checkpointed.

## Context

### ADK Context
- Documentation context: Based on TODOs in codebase.
- Reference context: None, Python is the target.
- General context: Separated independent task related to `Event.author` logic for later via the task queue.

### Language Specific Context
- Target language: Python
- Target repo: adk-python
- General context: Type hints and structured class architectures are strictly utilized. Be sure to filter `state_delta` contents before moving them to `event.actions.state_delta`. Note that when creating `EventActions(...)` in `_flush_output_and_deltas`, if the filtered state delta and the artifact delta are both empty, we may need to reconsider creating `event.actions` or assigning it `None`.

## Definition

### Data Models
No new models are needed.

### Inputs
`state_delta` dict within `ctx.actions.state_delta` inside `_NodeRunner`.

### Outputs
Modified behaviors in `_flush_deltas` and `_flush_output_and_deltas` that perform filters for keys starting with `"temp:"`.

### Side Effects
Event `state_delta` dicts will not contain keys that start with `"temp:"`.

## Constraints

### Invariants
The source `state_delta` from `ctx.actions` MUST still be cleared accurately via `state_delta.clear()` after filtering to prevent memory leaks and duplication on subsequent flushes.

### Preconditions
`ctx.actions.state_delta` contains keys pending flush.

### Postconditions
`event.actions.state_delta` does NOT contain keys with prefix `temp:`. `ctx.actions.state_delta` is fully cleared.

### Error Handling Protocols
Standard type checking, no custom errors.

### Breaking Change Analysis
No breaking changes. This satisfies a pending TODO for intended functionality.

### Testing

- #### Unit tests with >=95% New Line Coverage
Add unit tests ensuring `temp:some_key` is not propagated in `test_node_runner.py` for both `_flush_deltas` and `_flush_output_and_deltas`. Ensure valid keys are still persisted.
- #### Integration tests
N/A - Unit tests sufficient.
- #### Manual e2e test
N/A
