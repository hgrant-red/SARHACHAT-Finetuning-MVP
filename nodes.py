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

from config import UNIVERSAL_PERSONA

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
            chat_prompt = UNIVERSAL_PERSONA + (
                f"We still need to know: {', '.join(missing)}. "
                "Greet them and ask them conversationally for this information in 1-2 sentences."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)] + messages[-2:])
            return {"messages": [reply], "current_stage": 1, **state_updates}
        else:
            chat_prompt = (
                "Thank the user for sharing their profile info. "
                "Ask if there are any specific side effects they want to avoid, "
                "and what their preferred routine is with contraceptives (e.g., daily, monthly)."
                "keep this to a few sentences."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)] + messages[-2:])
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
                    SystemMessage(content="Strictly extract routine preferences and side effects for contraceptives. If a field is not explicitly mentioned, leave it empty."),
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
            chat_prompt = UNIVERSAL_PERSONA + (
                f"Ask the user about their: {', '.join(missing)}. "
                "Are they looking for something they take daily, monthly, or longer? Be concise and warm."
            )
            reply = llm.invoke([SystemMessage(content=chat_prompt)] + messages[-2:])
            return {"messages": [reply], "current_stage": 2, **state_updates}
        else:
            chat_prompt = UNIVERSAL_PERSONA + (
                "Acknowledge their preferences. Then, smoothly pivot to medical safety "
                "by asking this exact open-ended question: "
                "'Before we dive into specifics, do you have any diagnosed medical conditions, "
                "past surgeries, or take any daily medications?'"
                "Keep this to a few sentences."
            )

            reply = llm.invoke([SystemMessage(content=chat_prompt)] + messages[-2:])
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

        messages = state.get("messages", [])
        active_conditions = []
        retrieved_rules = []
        rag_context = "No major CDC contraindications found for these conditions."

        if vector_store:
            # 🛠️ CHANGE 2a: Updated condition mapping (thrombocytopenia and ≥ 35 years)
            condition_mapping = {
                "bleeding_disorder": "thrombocytopenia",
                "blood_clots": "DVT/PE",
                "high_blood_pressure": "hypertension",
                "over_35": "≥ 35 years", 
                "smoker": "Smoking",
                "migraines": "headache",
                "cancer": "cancer",
                "lupus": "Systemic lupus erythematosus"
            }

            # 🛠️ CHANGE 2b: Strict Metadata Filtering (k=30)
            for state_key, search_term in condition_mapping.items():
                if state.get(state_key) is True:
                    active_conditions.append(state_key)
                    print(f"🔍 [RAG] Filtering CDC Guidelines strictly for: {search_term}")
                    
                    docs = vector_store.similarity_search(
                        query="safety categories", 
                        k=30, 
                        filter={"condition": {"$ilike": f"%{search_term}%"}} 
                    )
                    
                    for doc in docs:
                        if doc.metadata.get("category_score", 1) >= 3:
                            retrieved_rules.append(doc.page_content)
            
            # 🛠️ CHANGE 2c: Fuzzy Semantic Search for Catch-All Conditions
            other_conditions = state.get("other_conditions", [])
            for condition in other_conditions:
                active_conditions.append(condition) # Add to list so the LLM prompt sees it
                print(f"🔍 [RAG] Performing fuzzy semantic search for volunteered condition: {condition}")
                
                # No strict metadata filter here, we let the embedding model find it semantically
                fuzzy_docs = vector_store.similarity_search(
                    query=f"safety categories for {condition}", 
                    k=15
                )
                
                for doc in fuzzy_docs:
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
        instruction = UNIVERSAL_PERSONA + (
            "Your task is to provide a birth control consultation summary. You must filter options "
            "based strictly on the CDC MEC context provided below, and then match the safe options to the patient's preferences.\n\n"
            
            "### LOGIC RULES\n"
            "1. **Safety First:** Any method listed in the CDC context below is Category 3 or 4 (UNSAFE). You must exclude them. "
            "If a method is NOT in the context, it is safe to recommend.\n"
            "2. **Preference Matching:** From the safe methods, recommend 2 to 3 options that fit their routine "
            "(e.g., 'daily' = pills; 'monthly/set-and-forget' = IUD, implant, or ring) while avoiding their unwanted side effects.\n\n"
            
            "### OUTPUT FORMAT\n"
            "Write a natural, consultative response to the patient in 2 or 3 paragraphs. Do NOT use structural headers like 'Recommendations' or 'Unsafe Methods'.\n"
            "- Start by gently explaining if any methods are medically unsafe for them and exactly *why* based on the CDC context.\n"
            "- Then, present their safe options, explaining how these fit their specific routine and side effect preferences.\n\n"
                        
            f"USER PREFERENCES:\n"
            f"- Preferred Routine/Delivery: {state.get('routine_preference')}\n"
            f"- Avoiding Side Effects: {', '.join(state.get('avoided_side_effects', []))}\n\n"
            
            f"ACTIVE HEALTH CONDITIONS: {', '.join(active_conditions) if active_conditions else 'None'}\n\n"
            
            f"--- CDC MEC STRICT WARNINGS (CONTEXT) ---\n{rag_context}\n-----------------------------------"
        )
        
        reply = llm.invoke([SystemMessage(content=instruction)] + messages[-2:])
        
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

        messages = state.get("messages", [])
        
        chat_prompt = UNIVERSAL_PERSONA + (
            "You just provided the user with their customized birth control recommendations. "
            "Ask them if they have any final questions about these options, or if they are ready "
            "for you to generate their final summary PDF for their doctor."
        )
        
        reply = llm.invoke([SystemMessage(content=chat_prompt)] + messages[-2:])
        return {"messages": [reply], "profile_verified": True, "current_stage": 5}
    return node
    
def build_stage_nodes(llm: BaseChatModel):
    return {
        "stage_1": _stage_1_node(llm),
        "stage_2": _stage_2_node(llm),
        "stage_4": _stage_4_node(llm),
        "stage_5": _stage_5_node(llm),
    }