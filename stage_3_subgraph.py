"""
Stage 3 Subgraph for SARHAchat.
Handles the multi-turn conversational chunking of the 8 CDC medical conditions.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from state import TriageState, Stage3Extraction

def build_stage_3_subgraph(llm: BaseChatModel):
    extractor = llm.with_structured_output(Stage3Extraction)

    def extract_health_info(state: TriageState) -> dict:
        print("\n⚙️ [SUBGRAPH] Running Stage 3 Extraction...")
        messages = state.get("messages", [])
        
        # FIX: Grab the last TWO messages (Bot's question + User's answer)
        # This gives the LLM context for words like "no", "none", or "this"
        recent_history = messages[-2:] if len(messages) >= 2 else messages

        extracted_data = {}
        # Only run extraction if the LAST message was from the user
        if messages and isinstance(messages[-1], HumanMessage):
            try:
                extraction_prompt = (
                    "You are a strict data extractor for CDC guidelines. "
                    "Read the AI's question and the User's answer to understand the context. "
                    "If the AI asked about specific conditions and the user says 'no', 'none', or 'no to all', "
                    "you MUST map 'False' to those specific conditions. "
                    "If a condition is not being discussed, leave it as null/None."
                )
                
                # We pass the SystemMessage PLUS the recent_history array
                extracted = extractor.invoke(
                    [SystemMessage(content=extraction_prompt)] + recent_history
                )
                
                for field in ["bleeding_disorder", "blood_clots", "high_blood_pressure", "over_35", "smoker", "migraines", "cancer", "lupus"]:
                    val = getattr(extracted, field)
                    if val is not None:
                        extracted_data[field] = val
            except Exception as e:
                print(f"Extraction failed safely: {e}")

        return extracted_data

    def assess_and_ask(state: TriageState) -> dict:
        print("\n⚙️ [SUBGRAPH] Assessing CDC Guidelines...")
        
        # We divide the 8 conditions into 3 specific chunks
        chunk_1 = ["over_35", "smoker"]
        chunk_2 = ["blood_clots", "high_blood_pressure", "bleeding_disorder"]
        chunk_3 = ["migraines", "cancer", "lupus"]
        
        friendly_names = {
            "over_35": "are you over the age of 35",
            "smoker": "do you smoke or vape",
            "blood_clots": "a history of blood clots",
            "high_blood_pressure": "high blood pressure",
            "bleeding_disorder": "a bleeding disorder",
            "migraines": "migraines",
            "cancer": "a history of cancer",
            "lupus": "Lupus"
        }

        # --- Check Chunk 1 (Age & Smoking) ---
        missing_1 = [c for c in chunk_1 if state.get(c) is None]
        if missing_1:
            friendly_missing = [friendly_names[c] for c in missing_1]
            chat_prompt = (
                "You are SARHAchat, a clinical assistant. "
                f"Ask the user: {', '.join(friendly_missing)}? "
                "CRITICAL: You MUST ask about all of these specific things. Keep it to one conversational sentence."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "health_screened": False, "current_stage": 3}

        # --- Check Chunk 2 (Blood/Cardio) ---
        missing_2 = [c for c in chunk_2 if state.get(c) is None]
        if missing_2:
            friendly_missing = [friendly_names[c] for c in missing_2]
            chat_prompt = (
                "You are SARHAchat. Thank them for the previous answers. "
                f"Now ask if they have any history of: {', '.join(friendly_missing)}. "
                "CRITICAL: You MUST list all of these specific conditions. Keep it brief."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "health_screened": False, "current_stage": 3}
            
        # --- Check Chunk 3 (Other conditions) ---
        missing_3 = [c for c in chunk_3 if state.get(c) is None]
        if missing_3:
            friendly_missing = [friendly_names[c] for c in missing_3]
            chat_prompt = (
                "You are SARHAchat. We are almost done with the health screening. "
                f"Ask if they have any history of: {', '.join(friendly_missing)}. "
                "CRITICAL: You MUST list all of these specific conditions."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)])
            return {"messages": [reply], "health_screened": False, "current_stage": 3}

        # --- All Checks Passed ---
        chat_prompt = (
            "You are SARHAchat. The health screening is complete. "
            "Thank the user warmly for sharing their medical history and let them know "
            "you are reviewing everything to provide a birth control recommendation next."
        )
        reply = llm.invoke([SystemMessage(content=chat_prompt)])
        return {"messages": [reply], "health_screened": True, "current_stage": 4}

    workflow = StateGraph(TriageState)
    workflow.add_node("extract_health_info", extract_health_info)
    workflow.add_node("assess_and_ask", assess_and_ask)
    
    workflow.add_edge(START, "extract_health_info")
    workflow.add_edge("extract_health_info", "assess_and_ask")
    workflow.add_edge("assess_and_ask", END)
    
    return workflow.compile()