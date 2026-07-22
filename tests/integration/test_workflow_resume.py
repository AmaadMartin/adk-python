import asyncio
import pytest
from google.adk.agents.context import Context
from google.adk.workflow._workflow import Workflow
from google.adk.workflow import node, START
from google.adk.events.request_input import RequestInput
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import InMemoryRunner

@node
async def node_a(ctx: Context, node_input: str) -> str:
    return node_input + " -> A"

@node(rerun_on_resume=True)
async def node_b(ctx: Context, node_input: str):
    print(f"NODE_B RESUME INPUTS: {ctx.resume_inputs}")
    if "req_b" not in ctx.resume_inputs:
        yield RequestInput(interrupt_id="req_b", message="Need B")
    else:
        res = ctx.resume_inputs.get("req_b", "NoResumeInput")
        if hasattr(res, "get"):
            res = res.get("result", res)
        yield node_input + f" -> B (with {res})"

@node
async def node_c(ctx: Context, node_input: str) -> str:
    print(f"NODE_C RESUME INPUTS: {ctx.resume_inputs}")
    return node_input + " -> C"

@pytest.mark.asyncio
async def test_workflow_resume_e2e():
    wf = Workflow(name="resume_e2e", rerun_on_resume=True, edges=[
        (START, node_a),
        (node_a, node_b),
        (node_b, node_c),
    ])
    
    app = App(
        name="resume_e2e_app",
        root_agent=wf,
        resumability_config=ResumabilityConfig(is_resumable=True)
    )
    runner = InMemoryRunner(app=app)
    
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai.types import Part, Content, FunctionResponse
    
    # Run 1
    session = await runner.session_service.create_session(app_name="resume_e2e_app", user_id="user1")
    user_event = Content(role="user", parts=[Part.from_text(text="Start")])
    events1 = [e async for e in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=user_event
    )]
    invocation_id = events1[0].invocation_id
    
    # Run 2: resume with input
    from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
    resume_event = Content(role="user", parts=[create_request_input_response(interrupt_id="req_b", response={"result": "HumanInput"})])
    events2 = [e async for e in runner.run_async(
        user_id="user1",
        session_id=session.id,
        invocation_id=invocation_id,
        new_message=resume_event
    )]
    
    # Verify final output event
    final_output = None
    for e in events2:
        if getattr(e, 'output', None) is not None:
            final_output = e.output
    
    print(f"Final output: {final_output}")
    assert final_output == "Start -> A -> B (with HumanInput) -> C"
    print("E2E Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_workflow_resume_e2e())
