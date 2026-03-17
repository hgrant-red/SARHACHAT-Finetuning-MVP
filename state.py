"""
State definitions for the SARHAchat 5-stage clinical triage.
TriageState (TypedDict) and Pydantic Unified models: extract + chat_reply in a single LLM call.
"""

from typing import TypedDict, List, Annotated, Literal
import operator

from pydantic import BaseModel, Field


# --- THE STATE (Clinical Stage Tracker) ---
class TriageState(TypedDict, total=False):
    """State dictionary for the LangGraph workflow."""

    # Chat History
    messages: Annotated[List[dict], operator.add]

    # Stage Tracker
    current_stage: int

    # Stage 1: Initial Information Gathering
    gender: str
    experience: str

    # Stage 2: Preference Screening
    frequency: str
    side_effects: List[str]

    # Stage 3: Health Screening
    medical_history: List[str]
    health_screened: bool  # True once we've completed the health screening (even if no conditions)

    # Stage 4 & 5: RAG Context & Final Outputs
    cdc_guideline_context: str
    recommendation: str
    profile_verified: bool


# --- Unified Pydantic models (extraction + chat_reply in one call) ---
class Stage1Unified(BaseModel):
    """Stage 1: extract gender & experience; respond in chat_reply."""

    gender: Literal["female", "male", "non-binary", "other", ""] = Field(
        default="",
        description="Strictly map to 'female', 'male', 'non-binary', or 'other'. Map to appropriate gende term if not easily mapped",
    )
    experience: str = Field(
        default="",
        description="Strictly 1-5 words summarizing past birth control use (e.g., 'used pill', 'none'). Empty if unknown.",
    )
    chat_reply: str = Field(
        description="Your empathetic reply to the user. Ask for missing info or say the transition phrase. Keep under 3 sentences.",
    )


class Stage2Unified(BaseModel):
    """Stage 2: extract frequency & side_effects; respond in chat_reply."""

    frequency: str = Field(
        default="",
        description="Strictly 1-3 words: how often they want to think about birth control (e.g., 'daily', 'monthly', 'set-and-forget'). Empty if not stated.",
    )
    side_effects: List[str] = Field(
        default_factory=list,
        description="Side effects they want to avoid.",
    )
    chat_reply: str = Field(
        description="Your empathetic reply. Ask for frequency and side-effect preferences, or thank them and transition to health screening.",
    )


class Stage3Unified(BaseModel):
    """Stage 3: extract medical_history; respond in chat_reply."""

    medical_history: List[str] = Field(
        default_factory=list,
        description="List of chronic medical conditions mentioned (e.g., migraines, hypertension, diabetes). Empty if none.",
    )

    answered_health_questions: bool = Field(
        description="True ONLY if the user explicitly listed medical conditions or explicitly stated they have none. False if they dodged the question, made a joke, or changed the subject."
    )
    chat_reply: str = Field(
        description="If answered_health_questions is True, thank them. If False, politely redirect and ask again if they have chronic medical conditions.",
    )


class Stage4Unified(BaseModel):
    """Stage 4: recommendation (RAG to be wired later); reply in chat_reply."""

    chat_reply: str = Field(
        description="Your brief recommendation or placeholder message based on preferences and health history.",
    )


class Stage5Unified(BaseModel):
    """Stage 5: profile verification (PDF stub); reply in chat_reply."""

    chat_reply: str = Field(
        description="Your reply: summarize profile and ask if everything looks correct before generating the summary.",
    )
