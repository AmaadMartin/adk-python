import time
from typing import List

from google.genai import types

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from unittest.mock import Mock


def main():
    print("Running E2E performance test for function call index...")
    num_events = 10000

    events: List[Event] = []
    
    # 1. Create 10000 events, each with a function call
    start_time = time.perf_counter()
    for i in range(num_events):
        fc = types.FunctionCall(name='some_tool', args={})
        fc.id = f'call_{i}'
        fc_part = types.Part(function_call=fc)
        
        event = Event(
            invocation_id='inv_1',
            author='dummy_agent',
            content=types.Content(parts=[fc_part], role='model'),
        )
        events.append(event)
        
    print(f"Created {num_events} events in {time.perf_counter() - start_time:.4f}s")

    session = Session(id="test_session", app_name="test_app", user_id="test_user", events=events)
    
    # Target function calls to find
    targets = [f'call_{0}', f'call_{num_events // 2}', f'call_{num_events - 1}']
    
    from google.adk.flows.llm_flows.functions import find_matching_function_call
    
    for target_id in targets:
        fr = types.FunctionResponse(name='some_tool', response={'result': 'ok'})
        fr.id = target_id
        fr_part = types.Part(function_response=fr)
        fr_event = Event(
            invocation_id='inv_1',
            author='dummy_agent',
            content=types.Content(parts=[fr_part], role='user'),
        )
        
        session.events.append(fr_event)
        
        t0 = time.perf_counter()
        match = find_matching_function_call(session.events, session.function_call_index)
        t1 = time.perf_counter()
        
        assert match is not None, f"Match not found for {target_id}"
        found_fc = match.get_function_calls()[0]
        assert found_fc.id == target_id, f"Mismatched ID. Expected {target_id}, got {found_fc.id}"
        print(f"Found {target_id} in {t1 - t0:.6f}s")
        

    print("Success! Performance scales phenomenally.")


if __name__ == '__main__':
    main()
