## Task

### User intent with respect to ADK
Refactor Python's `find_matching_function_call` and `find_event_by_function_call_id` functions to use an index for O(1) lookups instead of O(N) backward searching. This brings Python's performance semantics to parity with an expected `adk-js` refactor.

### Feature Description
Currently, `find_event_by_function_call_id` in `adk-python/src/google/adk/flows/llm_flows/functions.py` iterates backwards over the `events` list to find the matching function call for a function response. 
The plan is to introduce a mapped index (e.g., `dict[str, Event]` where the key is the `function_call_id`) that is maintained as events are added via `InvocationContext` and `Session`, allowing `find_event_by_function_call_id` to perform lookup in O(1) time.

### Use Cases & Examples
- Long-running sessions with many agent iterations and function calls currently incur O(N) lookup penalties for every function response. This refactoring ensures constant time resolution regardless of session scale.

## Context

### ADK Context
- Documentation context: `InvocationContext` and runners currently assume a list of `events` is enough, but maintaining state efficiently is paramount.
- Reference context: Parity request with `adk-js` implies aligning the core function logic so it doesn't do a manual iterative search through `events` backwards.
- General context: Functions to update include `find_matching_function_call` and `find_event_by_function_call_id`. These are utilized in `remote_a2a_agent.py`, `llm_agent.py`, `invocation_context.py`, and `runners.py`.

### Language Specific Context
- Target language: Python
- Target repo: `adk-python`
- General context: Python `dict` provides O(1) time complexity for fetching values by key.

## Definition

### Data Models 
Add an internal index `_function_call_index: dict[str, Event]` parameter (or context state variable) that maps `function_call.id` to its parent `Event`.

### Inputs
Function signatures for `find_matching_function_call` and `find_event_by_function_call_id` will need to accept the index (or access it transparently via context classes if moved into them).

### Outputs
Returns `Optional[Event]` in O(1).

### Side Effects
The indexing dictionary needs to be kept in sync with `events` arrays.

## Constraints

### Invariants
The dictionary must contain keys for all function calls present in the session history.

### Preconditions
When an event containing `function_calls` is processed or appended, the index dictionary must be updated.

### Postconditions
The lookup function matches the same event it successfully matched previously but without looping.

### Error Handling Protocols
If a `function_call_id` is missing in the index, standard `None` returns applies.

### Breaking Change Analysis
This is an internal refactoring for efficiency; if `find_matching_function_call` signature changes, all internal call sites must be updated accordingly.

### Testing

- #### Unit tests with >=95% New Line Coverage
Ensure existing tests in `test_invocation_context.py` and `test_functions_simple.py` continue to pass.
- #### Integration tests
Agent flows with multiple rounds of tool use (e.g., `test_remote_a2a_agent.py`) must be fully validated.
- #### Manual e2e test
A test with hundreds of tool calls confirming performance scaling.
