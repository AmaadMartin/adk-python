from typing import Any
import pytest
from google.genai import types
from google.adk.events.event import Event
from google.adk.sessions.session import Session


def test_function_call_index_incremental():
    """Tests that function calls are incrementally indexed."""
    session = Session(id="1", app_name="test", user_id="user")
    
    # 1. Add event with function call
    fc = types.FunctionCall(name='test_tool', args={})
    fc.id = "call_1"
    event1 = Event(
        invocation_id="inv1",
        author="agent",
        content=types.Content(parts=[types.Part(function_call=fc)], role="model")
    )
    session.events.append(event1)
    
    assert session.function_call_index == {"call_1": event1}
    assert session._indexed_events_count == 1
    
    # 2. Add another event
    fc2 = types.FunctionCall(name='test_tool', args={})
    fc2.id = "call_2"
    event2 = Event(
        invocation_id="inv1",
        author="agent",
        content=types.Content(parts=[types.Part(function_call=fc2)], role="model")
    )
    session.events.append(event2)
    
    assert session.function_call_index == {"call_1": event1, "call_2": event2}
    assert session._indexed_events_count == 2
    
def test_function_call_index_truncation():
    """Tests that the index rebuilds when events are truncated."""
    session = Session(id="1", app_name="test", user_id="user")
    
    fc = types.FunctionCall(name='test_tool', args={})
    fc.id = "call_1"
    event1 = Event(
        invocation_id="inv1",
        author="agent",
        content=types.Content(parts=[types.Part(function_call=fc)], role="model")
    )
    session.events.append(event1)
    
    # Add a second event
    fc2 = types.FunctionCall(name='test_tool', args={})
    fc2.id = "call_2"
    event2 = Event(
        invocation_id="inv1",
        author="agent",
        content=types.Content(parts=[types.Part(function_call=fc2)], role="model")
    )
    session.events.append(event2)
    
    # Access index to cache it
    _ = session.function_call_index
    assert session._indexed_events_count == 2
    
    # Remove one event to simulate truncation
    session.events.pop()
    
    # Index should rebuild for remaining events
    assert session.function_call_index == {"call_1": event1}
    assert session._indexed_events_count == 1

def test_get_matching_function_call():
    """Tests getting a matching function call based on a function response."""
    session = Session(id="1", app_name="test", user_id="user")
    
    # Test empty events
    assert session.get_matching_function_call() is None
    
    # 1. Add event with function call
    fc = types.FunctionCall(name='test_tool', args={})
    fc.id = "call_1"
    event1 = Event(
        invocation_id="inv1",
        author="agent",
        content=types.Content(parts=[types.Part(function_call=fc)], role="model")
    )
    session.events.append(event1)
    
    # Test last event has no function response
    assert session.get_matching_function_call() is None
    
    # 2. Add event with function response
    fr = types.FunctionResponse(name='test_tool', response={"status": "ok"})
    fr.id = "call_1"
    event2 = Event(
        invocation_id="inv1",
        author="user",
        content=types.Content(parts=[types.Part(function_response=fr)], role="user")
    )
    session.events.append(event2)
    
    # Test it finds the function call
    assert session.get_matching_function_call() == event1
    
    # 3. Add response with no ID
    fr2 = types.FunctionResponse(name='test_tool', response={"status": "ok"})
    fr2.id = ""
    event3 = Event(
        invocation_id="inv1",
        author="user",
        content=types.Content(parts=[types.Part(function_response=fr2)], role="user")
    )
    session.events.append(event3)
    assert session.get_matching_function_call() is None
