"""
Dynamic router: routes to the appropriate stage based on exact state variables.
Uses the same fields as TriageState (state.py). Each node returns to END after replying.
"""

from langgraph.graph import END

from state import TriageState


def dynamic_router(state: TriageState):
    """
    Route using the exact variables defined in state.py.
    Order of checks matters: we advance only when prior stage data is complete.
    """
    # Stage 1: Initial Information Gathering (gender, experience)
    if not state.get("gender") or not state.get("experience"):
        return "stage_1"

    # Stage 2: Preference Screening (frequency)
    if not state.get("frequency"):
        return "stage_2"

    # Stage 3: Health Screening
    # Stay in stage_3 until health_screened is True (set when user answers; medical_history may be empty)
    if not state.get("health_screened", False):
        return "stage_3"

    # Otherwise -> Stage 4 (recommendation)
    if not state.get("recommendation"):
        return "stage_4"

    # Stage 5: Profile verification
    if not state.get("profile_verified"):
        return "stage_5"

    return END
