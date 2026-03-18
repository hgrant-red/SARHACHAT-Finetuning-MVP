"""
State definitions for the SARHAchat 5-stage clinical triage.
"""

from typing import TypedDict, List, Annotated, Optional
import operator
from pydantic import BaseModel, Field

# --- THE STATE (Clinical Stage Tracker) ---
class TriageState(TypedDict, total=False):
    """State dictionary for the LangGraph workflow. Shared across main graph and subgraphs."""

    messages: Annotated[List[dict], operator.add]
    current_stage: int

    # Stage 1: Initial Information Gathering (Trimmed)
    pronouns: str 
    pregnancy_plans: str
    experience: str

    # Stage 2: Preference Screening (Trimmed & Combined)
    routine_preference: str  # Combines frequency and delivery
    avoided_side_effects: List[str]
    preferences_screened: bool

    # Stage 3: Health Screening (explicit boolean flags for CDC MEC Guidelines)
    bleeding_disorder: Optional[bool]
    blood_clots: Optional[bool]
    high_blood_pressure: Optional[bool]
    over_35: Optional[bool]
    smoker: Optional[bool]
    migraines: Optional[bool]
    cancer: Optional[bool]
    lupus: Optional[bool]
    health_screened: bool 

    # Stage 4 & 5: Completion
    recommendation: str
    profile_verified: bool


# --- EXTRACTION ONLY Pydantic Models ---

class Stage1Extraction(BaseModel):
    pronouns: str = Field(
        default="",
        description="Extract the user's pronouns if mentioned (e.g., 'she/her', 'they/them'). Leave empty if not stated.",
    )
    pregnancy_plans: str = Field(
        default="",
        description="Extract future pregnancy plans (e.g., 'never', 'in a few years'). Leave empty if not stated.",
    )
    experience: str = Field(
        default="",
        description="Strictly 1-5 words summarizing past birth control use (e.g., 'used pill', 'none'). Leave empty if not stated.",
    )

class Stage2Extraction(BaseModel):
    routine_preference: str = Field(
        default="",
        description="Extract their preferred routine or delivery method (e.g., 'daily pill', 'set-and-forget', 'IUD', 'monthly'). Leave empty if not stated.",
    )
    avoided_side_effects: List[str] = Field(
        default_factory=list,
        description="List of side effects they explicitly want to AVOID (e.g., 'weight gain', 'acne'). Empty if none stated.",
    )

class Stage3Extraction(BaseModel):
    bleeding_disorder: Optional[bool] = Field(default=None, description="True if bleeding disorder. False if explicitly denied.")
    blood_clots: Optional[bool] = Field(default=None, description="True if blood clots. False if explicitly denied.")
    high_blood_pressure: Optional[bool] = Field(default=None, description="True if high blood pressure. False if explicitly denied.")
    over_35: Optional[bool] = Field(default=None, description="True if over 35. False if explicitly denied.")
    smoker: Optional[bool] = Field(default=None, description="True if smoker. False if explicitly denied.")
    migraines: Optional[bool] = Field(default=None, description="True if migraines. False if explicitly denied.")
    cancer: Optional[bool] = Field(default=None, description="True if history of breast/liver/cervical cancer. False if explicitly denied.")
    lupus: Optional[bool] = Field(default=None, description="True if Lupus (SLE). False if explicitly denied.")