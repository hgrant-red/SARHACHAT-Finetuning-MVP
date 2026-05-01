"""
SARHAchat MVP entry point.
OpenShift vLLM connection setup and interactive consultation loop.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from state import TriageState
from graph import compile_app



# OpenShift internal vLLM URL 
OPENSHIFT_MODEL_URL = "---ENTER MODEL URL HERE FROM MODEL DEPLOYMENTS---"
os.environ["OPENAI_API_BASE"] = OPENSHIFT_MODEL_URL

# LLM for all nodes
llm = ChatOpenAI(
    model="sarhachat",  #Alternate calling base model vs LORA adaptor model version by changing model name
    # model="sarhachat-dynamic2",  
    temperature=0.2,
    max_tokens=1500,
)

def get_initial_state() -> TriageState:
    """Blank patient state for starting a new consultation."""
    return {
        "current_stage": 1,
        "messages": [],
        "pronouns": "",
        "pregnancy_plans": "",
        "experience": "",
        "routine_preference": "",
        "avoided_side_effects": [],
        "preferences_screened": False,
        "bleeding_disorder": None,
        "blood_clots": None,
        "high_blood_pressure": None,
        "over_35": None,
        "smoker": None,
        "migraines": None,
        "cancer": None,
        "lupus": None,
        "other_conditions": [],
        "health_screened": False,
        "recommendation": "",
        "profile_verified": False,
    }

# --- DASHBOARD FUNCTIONS ---
def fmt(val):
    """Helper to format boolean/None values for the dashboard."""
    if val is None: return "[ ]"
    return str(val)

def fmt_list(val_list):
    """Helper to format lists for the dashboard."""
    if not val_list: return "[ ]"
    return ", ".join(val_list)

def print_state_tracker(state: TriageState):
    """Prints the static clinical dashboard based on current state."""
    print(f"\n{'='*60}")
    print(f"🏥 SARHAchat State Dashboard | Current Stage: {state.get('current_stage', 1)}")
    print(f"{'='*60}")
    
    print("--- Stage 1: Profile ---")
    print(f" Pronouns:       {state.get('pronouns') or '[ ]'}")
    print(f" Pregnancy:      {state.get('pregnancy_plans') or '[ ]'}")
    print(f" Experience:     {state.get('experience') or '[ ]'}\n")

    print(f"--- Stage 2: Preferences (Screened: {state.get('preferences_screened', False)}) ---")
    print(f" Routine:        {state.get('routine_preference') or '[ ]'}")
    print(f" Avoiding:       {fmt_list(state.get('avoided_side_effects'))}\n")

    print(f"--- Stage 3: Health Risks (Screened: {state.get('health_screened', False)}) ---")
    print(f" > 35: {fmt(state.get('over_35'))}    | Smoker: {fmt(state.get('smoker'))}   | High BP: {fmt(state.get('high_blood_pressure'))}")
    print(f" Clots: {fmt(state.get('blood_clots'))} | Bleeding: {fmt(state.get('bleeding_disorder'))} | Migraines: {fmt(state.get('migraines'))}")
    print(f" Cancer: {fmt(state.get('cancer'))} | Lupus: {fmt(state.get('lupus'))}\n")
    print(f"Other: {', '.join(state.get('other_conditions', [])) if state.get('other_conditions') else '[ ]'}")

    print("--- Stage 4 & 5: Completion ---")
    rec_status = "Generated" if state.get('recommendation') else "[ ]"
    print(f" Recommendation: {rec_status}")
    print(f" Verified:       {state.get('profile_verified', False)}")
    print(f"{'='*60}\n")
# ------------------------------------------

def main() -> None:
    print(f"🔗 LLM: {OPENSHIFT_MODEL_URL}\n")
    app = compile_app(llm)
    print("✅ LangGraph compiled.\n")

    # Define the session ID for LangGraph MemorySaver
    config = {"configurable": {"thread_id": "patient_123"}}

    print("🚀 Starting SARHAchat Consultation Mode (Type 'quit' to exit)\n")

    # First invocation: Seed the memory with our blank initial state
    result = app.invoke(get_initial_state(), config=config)
    print(f"🤖 SARHAchat: {result['messages'][-1].content}")
    
    # Print the empty dashboard on startup
    print_state_tracker(result)

    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        # 💥 Because of MemorySaver, we ONLY need to pass the new message!
        # LangGraph will pull the rest of the state automatically.
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        print(f"\n🤖 SARHAchat: {result['messages'][-1].content}")
        
        # Display the updated static clinical dashboard
        print_state_tracker(result)

if __name__ == "__main__":
    main()