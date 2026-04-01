"""
LangGraph nodes for SARHAchat clinical triage.
Returns standard dictionaries for guaranteed state updates.
"""
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END

from state import (
    TriageState,
    Stage1Extraction,
    Stage2Extraction
)

def _stage_1_node(llm: BaseChatModel):
    # [Your existing Stage 1 code remains exactly the same]
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
    # [Your existing Stage 2 code remains exactly the same]
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
    # Initialize the Vector Store ONCE when the app compiles
    db_url = os.environ.get("DATABASE_URL")
    vector_store = None
    if db_url:
        print("⚙️ [SYSTEM] Initializing PGVector connection for Stage 4...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name="cdc_mec_rules",
            connection=db_url,
        )
    else:
        print("⚠️ [WARNING] DATABASE_URL not found. RAG will be disabled in Stage 4.")

    def node(state: TriageState) -> dict:
        print("\n⚙️ [SYSTEM] Running Stage 4 Logic (Structured RAG)...")
        
        active_conditions = []
        retrieved_rules = []
        
        # 1. Map the state booleans to the exact keywords found in the CDC tables
        condition_mapping = {
            "bleeding_disorder": "bleeding",
            "blood_clots": "dvt",
            "high_blood_pressure": "hypertension",
            "over_35": "35",
            "smoker": "smoking",
            "migraines": "migraine",
            "cancer": "cancer",
            "lupus": "sle" # Systemic lupus erythematosus
        }

        rag_context = "No major CDC contraindications found for these conditions."

        # 2. Query PGVector using Strict Metadata Filtering
        if vector_store:
            for state_key, search_term in condition_mapping.items():
                if state.get(state_key) is True:
                    active_conditions.append(state_key)
                    print(f"🔍 [RAG] Filtering CDC Guidelines strictly for: {search_term}")
                    
                    # 💥 Structured RAG: We bypass semantic guessing and filter strictly by the metadata column
                    docs = vector_store.similarity_search(
                        query="safety categories", 
                        k=15, 
                        filter={"condition": {"$ilike": f"%{search_term}%"}} 
                    )
                    
                    for doc in docs:
                        # Only keep the chunks where the CDC strictly prohibits or warns against it (Categories 3 & 4)
                        if doc.metadata.get("category_score", 1) >= 3:
                            retrieved_rules.append(doc.page_content)
            
            # Remove duplicates and combine into our final context
            if retrieved_rules:
                rag_context = "\n".join(set(retrieved_rules))
                
                # --- DEBUGGING BLOCK ---
                print("\n" + "▼"*60)
                print(f"🛠️ DEBUG: METADATA FILTERED CHUNKS (Categories 3 & 4)")
                print("▼"*60)
                print(rag_context)
                print("▲"*60 + "\n")

        # 3. The "Separation of Concerns" Prompt
        instruction = (
            "You are SARHAchat, a clinical birth control assistant in Stage 4: Recommendation. "
            "Your job is to recommend the best contraceptive methods by cross-referencing the user's preferences "
            "with the strict medical safety rules in the CDC MEC Guidelines provided below.\n\n"
            
            "--- STEP 1: SAFETY FIRST (STRICT RULES) ---\n"
            "Review the CDC MEC Guidelines below. If a method is listed as Category 3 or Category 4 "
            "for ANY of the user's active health conditions, it is medically unsafe. You MUST NOT recommend it. "
            "Instead, briefly explain why it is unsafe.\n\n"
            
            "--- STEP 2: APPLY PREFERENCES ---\n"
            "Out of the methods that are medically safe (Category 1 or 2), prioritize recommending the ones "
            "that best match the user's routine and side-effect preferences.\n\n"
            
            f"USER PREFERENCES:\n"
            f"- Preferred Routine/Delivery: {state.get('routine_preference')}\n"
            f"- Avoiding Side Effects: {', '.join(state.get('avoided_side_effects', []))}\n\n"
            
            f"ACTIVE HEALTH CONDITIONS: {', '.join(active_conditions) if active_conditions else 'None'}\n\n"
            
            f"--- CDC MEC GUIDELINES (CONTEXT) ---\n{rag_context}\n-----------------------------------"
        )
        
        reply = llm.invoke([SystemMessage(content=instruction)])
        
        return {
            "messages": [reply], 
            "recommendation": reply.content, 
            "current_stage": 5
        }
    return node

def _stage_5_node(llm: BaseChatModel):
    # [Your existing Stage 5 code remains exactly the same]
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