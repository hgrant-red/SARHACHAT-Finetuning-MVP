"""
LangGraph workflow: wire nodes, router, and compile the app.
Each node runs once per invocation then edges to END to wait for user input.
"""

from langgraph.graph import StateGraph, START, END

from state import TriageState
from router import dynamic_router
from nodes import (
    stage_1_initial_info,
    stage_2_preferences,
    stage_3_health_screening,
    stage_4_recommendation,
    stage_5_profile_verification,
)


def compile_app(llm):
    """
    Build and compile the SARHAchat LangGraph app.
    llm: ChatOpenAI (or compatible) instance for OpenShift vLLM.
    """
    workflow = StateGraph(TriageState)

    # Bind llm to each node so they don't import it
    workflow.add_node("stage_1", lambda s: stage_1_initial_info(s, llm))
    workflow.add_node("stage_2", lambda s: stage_2_preferences(s, llm))
    workflow.add_node("stage_3", lambda s: stage_3_health_screening(s, llm))
    workflow.add_node("stage_4", lambda s: stage_4_recommendation(s, llm))
    workflow.add_node("stage_5", lambda s: stage_5_profile_verification(s, llm))

    # Router decides where to send each new invocation (based on state)
    workflow.add_conditional_edges(START, dynamic_router)

    # Every node goes to END after generating a response (no infinite loop)
    workflow.add_edge("stage_1", END)
    workflow.add_edge("stage_2", END)
    workflow.add_edge("stage_3", END)
    workflow.add_edge("stage_4", END)
    workflow.add_edge("stage_5", END)

    return workflow.compile()
