from langgraph.pregel import Pregel

from simple_agent.graph import graph, read_preference, update_preference, utc_now


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_utc_now_tool() -> None:
    result = utc_now.invoke({})
    assert isinstance(result, str)
    assert "T" in result
