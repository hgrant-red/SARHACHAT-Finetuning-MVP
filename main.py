"""
SARHAchat MVP entry point.
OpenShift vLLM connection setup and interactive consultation loop.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from state import TriageState
from graph import compile_app

# OpenShift internal vLLM URL (do not remove or overwrite)
OPENSHIFT_MODEL_URL = "https://redhataillama-31-8b-instruct-predictor.sarhachat-mvp.svc.cluster.local:8443/v1"
os.environ["OPENAI_API_BASE"] = OPENSHIFT_MODEL_URL

# LLM for all nodes: redhataillama on vLLM, low temperature for clinical safety
llm = ChatOpenAI(
    model="redhataillama-31-8b-instruct",
    temperature=0.2,
    max_tokens=1500,
)


def get_initial_state() -> TriageState:
    """Blank patient state for starting a new consultation."""
    return {
        "current_stage": 1,
        "messages": [],
        "gender": "",
        "experience": "",
        "frequency": "",
        "side_effects": [],
        "medical_history": [],
        "health_screened": False,
        "cdc_guideline_context": "",
        "recommendation": "",
        "profile_verified": False,
    }


def main() -> None:
    print(f"🔗 LLM: {OPENSHIFT_MODEL_URL}\n")
    app = compile_app(llm)
    print("✅ LangGraph compiled.\n")

    patient_state: TriageState = get_initial_state()
    print("🚀 Starting SARHAchat Consultation Mode (Type 'quit' to exit)\n")

    # First invocation: bot greeting
    result = app.invoke(patient_state)
    patient_state = result
    print(f"🤖 SARHAchat: {patient_state['messages'][-1].content}")

    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        patient_state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(patient_state)
        patient_state = result

        print(f"\n🤖 SARHAchat: {patient_state['messages'][-1].content}")
        print(
            f"📊 [STATE TRACKER] Stage: {patient_state.get('current_stage')} | "
            f"Gender: {patient_state.get('gender')} | Exp: {patient_state.get('experience')} | "
            f"Freq: {patient_state.get('frequency')} | "
            f"Health: {patient_state.get('medical_history')} (Screened: {patient_state.get('health_screened')})"
        )


if __name__ == "__main__":
    main()
