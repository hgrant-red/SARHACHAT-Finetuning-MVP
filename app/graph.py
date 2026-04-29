"""
LangGraph workflow: compiles the 5-stage triage with Subgraph integration.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import TriageState
from nodes import build_stage_nodes
from stage_3_subgraph import build_stage_3_subgraph

def route_from_start(state: TriageState):
    """Router dictates where to resume based on current state."""
    stage = state.get("current_stage", 1)
    return f"stage_{stage}"

def compile_app(llm):
    workflow = StateGraph(TriageState)
    
    nodes = build_stage_nodes(llm)
    stage_3_subgraph = build_stage_3_subgraph(llm)

    workflow.add_node("stage_1", nodes["stage_1"])
    workflow.add_node("stage_2", nodes["stage_2"])
    workflow.add_node("stage_3", stage_3_subgraph) 
    workflow.add_node("stage_4", nodes["stage_4"])
    workflow.add_node("stage_5", nodes["stage_5"])

    # Resume the graph at whatever stage the state tracker is currently on
    workflow.add_conditional_edges(START, route_from_start)

    # 💥 GUARANTEE that the graph halts and waits for the user after EVERY turn
    workflow.add_edge("stage_1", END)
    workflow.add_edge("stage_2", END)
    workflow.add_edge("stage_3", END)
    workflow.add_edge("stage_4", END)
    workflow.add_edge("stage_5", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)