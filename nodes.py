"""
LangGraph nodes for SARHAchat clinical triage.
Returns standard dictionaries for guaranteed state updates.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END

from state import (
    TriageState,
    Stage1Extraction,
    Stage2Extraction
)

def _stage_1_node(llm: BaseChatModel):
    extractor = llm.with_structured_output(Stage1Extraction)

    def node(state: TriageState) -> dict:
        print("\n⚙️ [SYSTEM] Running Stage 1 Logic...")
        messages = state.get("messages", [])
        last_user_msg = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""

        pronouns = state.get("pronouns", "")
        pregnancy = state.get("pregnancy_plans", "")
        experience = state.get("experience", "")

        if last_user_msg:
            try:
                extracted = extractor.invoke([
                    SystemMessage(content="Strictly extract data. If a field is not explicitly mentioned by the user, you MUST leave it empty. Do not guess or assume 'no'."),
                    HumanMessage(content=last_user_msg)
                ])
                pronouns = pronouns or extracted.pronouns
                pregnancy = pregnancy or extracted.pregnancy_plans
                experience = experience or extracted.experience
            except Exception as e:
                print(f"Extraction failed safely: {e}")

        missing = []
        if not pronouns: missing.append("pronouns")
        if not pregnancy: missing.append("future pregnancy plans")
        if not experience: missing.append("prior experience with birth control")

        state_updates = {
            "pronouns": pronouns,
            "pregnancy_plans": pregnancy,
            "experience": experience
        }

        if missing:
            chat_prompt = (
                "You are SARHAchat, an empathetic clinical assistant. "
                f"We still need to know the following about the user: {', '.join(missing)}. "
                "Ask them conversationally and warmly for this information in 1-2 sentences."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "current_stage": 1, **state_updates}
        else:
            chat_prompt = (
                "You are SARHAchat. You just gathered their profile info. "
                "Thank them warmly and tell them we are going to talk about their birth control preferences next."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "current_stage": 2, **state_updates}

    return node

def _stage_2_node(llm: BaseChatModel):
    extractor = llm.with_structured_output(Stage2Extraction)

    def node(state: TriageState) -> dict:
        print("\n⚙️ [SYSTEM] Running Stage 2 Logic...")
        messages = state.get("messages", [])
        last_user_msg = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""

        routine = state.get("routine_preference", "")
        avoided = state.get("avoided_side_effects", [])

        if last_user_msg:
            try:
                extracted = extractor.invoke([
                    SystemMessage(content="Strictly extract preferences. If a field is not explicitly mentioned, leave it empty."),
                    HumanMessage(content=last_user_msg)
                ])
                routine = routine or extracted.routine_preference
                avoided = list(set(avoided + extracted.avoided_side_effects))
            except Exception as e:
                print(f"Extraction failed safely: {e}")

        missing = []
        if not routine: missing.append("preferred routine or delivery method (e.g., daily pill, set-and-forget IUD)")

        state_updates = {
            "routine_preference": routine,
            "avoided_side_effects": avoided
        }

        if missing:
            chat_prompt = (
                "You are SARHAchat. We are discussing contraceptive preferences for frequency of use or delivery method. "
                f"Ask the user about their: {', '.join(missing)}. "
                "Are they looking for something they take daily, monthly, or longer? Do you prefer a pill, patch, iud or something else? Be concise and warm."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "current_stage": 2, **state_updates}
        else:
            chat_prompt = (
                "You are SARHAchat, a clinical birth control assistant. "
                "Thank the user for sharing their preferences. "
                "Tell them that to make safe recommendations, we need to do a quick medical health screening next. "
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "preferences_screened": True, "current_stage": 3, **state_updates}

    return node

def _stage_4_node(llm: BaseChatModel):
    def node(state: TriageState) -> dict:
        print("\n⚙️ [SYSTEM] Running Stage 4 Logic...")
        chat_prompt = (
            "You are SARHAchat, a clinical birth control assistant. "
            "The user has completed their profile and medical screening. "
            "Provide a brief, contraceptive recommendation based on what they have shared."
        )
        reply = llm.invoke([SystemMessage(content=chat_prompt)])
        return {"messages": [reply], "recommendation": reply.content, "current_stage": 5}
    return node

def _stage_5_node(llm: BaseChatModel):
    def node(state: TriageState) -> dict:
        print("\n⚙️ [SYSTEM] Running Stage 5 Logic...")
        chat_prompt = "You are SARHAchat. Ask if everything looks correct."
        reply = llm.invoke([SystemMessage(content=chat_prompt)])
        return {"messages": [reply], "profile_verified": True, "current_stage": 5}
    return node

def build_stage_nodes(llm: BaseChatModel):
    return {
        "stage_1": _stage_1_node(llm),
        "stage_2": _stage_2_node(llm),
        "stage_4": _stage_4_node(llm),
        "stage_5": _stage_5_node(llm),
    }