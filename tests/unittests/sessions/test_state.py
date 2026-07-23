from google.adk.sessions.state import State

def test_state_mapping_methods():
    state = State(value={"a": 1, "b": 2}, delta={})
    
    assert len(state) == 2
    
    keys = list(iter(state))
    assert keys == ["a", "b"]
    
    del state["a"]
    assert "a" not in state
    assert len(state) == 1
    
    state.update({"c": 3})
    assert state["c"] == 3
    assert len(state) == 2
    
    # testing iterating over a combination of value and delta
    state = State(value={"a": 1}, delta={"b": 2})
    assert len(state) == 2
    assert sorted(list(iter(state))) == ["a", "b"]
    
    del state["a"]
    assert "a" not in state
    assert len(state) == 1
    assert "a" not in state._value and "a" not in state._delta
    
    # delete from delta
    del state["b"]
    assert "b" not in state
    assert len(state) == 0
