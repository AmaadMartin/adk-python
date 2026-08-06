## Task

### User intent with respect to ADK
Enable `Workflow` in `adk-python` to properly resume its internal state (`loop_state`) from the latest available checkpoint event when initialized in a resumption context.

### Feature Description
In `src/google/adk/workflow/_workflow.py`, `Workflow._run_impl()` creates a new `_LoopState` and delegates to `replay_mgr.scan_workflow_events(ctx)`. However, the current code is missing the logic to parse the `agent_state` checkpoint (which contains the snapshot of all static `loop_state.nodes` and their statuses). By implementing a loop through `ic.session.events`, we can identify the latest checkpoint authored by `self.name` where `actions.agent_state` is present. If found, we rehydrate `loop_state.nodes` and accumulate `loop_state.interrupt_ids` so the orchestration loop accurately starts from where it left off, rather than re-triggering from START or incorrectly falling back to just historical replay logic.

### Use Cases & Examples
- Resuming a workflow after a human-in-the-loop (HITL) interrupt. The workflow parses the checkpoint, sees a node is in `WAITING` with unhandled `interrupt_ids`, and continues smoothly by absorbing incoming user responses via `ctx.resume_inputs`.

## Context

### ADK Context
- Reference context: `loop_state.nodes` captures a state representation like `{"node_name": {"status": "WAITING", "interrupts": ["hitl_123"]}}`.
- Workflow orchestration uses `loop_state.sequence_barrier` and `loop_state.recovered_executions`, but the actual graph topology triggers depend on `loop_state.nodes` state.

### Language Specific Context
- Target language: Python
- Target repo: `adk-python`
- General context: The `_LoopState` class in `_workflow.py` maintains `nodes: dict[str, NodeState]`. `NodeState` is a Pydantic `BaseModel` from `_node_state.py`.

## Definition

### Data Models 
- `NodeState` (from `_node_state.py`): Pydantic model representing node state.
- `EventActions.agent_state` (from `event_actions.py`): Contains the nodes dictionary from a checkpoint.

### Inputs
`ctx._invocation_context.session.events` containing `Event` objects. We scan for an event matching:
- `event.author == self.name`
- `event.actions is not None and event.actions.agent_state is not None`

### Outputs
- Hydrated `loop_state.nodes` using `NodeState.model_validate` or `.parse_obj`.
- Updates to `loop_state.interrupt_ids` via `.update(node.interrupts)`.

### Side Effects
Hydrates `loop_state` properly in resumable contexts.

## Constraints

### Invariants
- Must pick the *latest* checkpoint event.

### Preconditions
- `ctx._invocation_context` must be properly initialized with session events.

### Postconditions
- `loop_state.nodes` reflects the last saved state before the workflow was stopped / interrupted.

### Error Handling Protocols
- Missing `nodes` key in `agent_state` should be handled gracefully (e.g., skip or default to empty).

### Breaking Change Analysis
- Non-breaking. This fixes a `TODO` for existing unimplemented behavior without changing existing APIs.

### Testing

- #### Unit tests with >=95% New Line Coverage
  Add tests in `tests/unittests/workflow/test_workflow.py` validating that when `session.events` contains a checkpoint, it is correctly parsed into `loop_state.nodes`.
- #### Integration tests
  `test_workflow_hitl.py` has tests matching checkpoint behavior (`test_workflow_hitl.py` lines 149 onwards). Check these to ensure they pass once implemented.
- #### Manual e2e test
  N/A
