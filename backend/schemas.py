"""
Pydantic data schemas for Citizen Profile Request and Prediction Response.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CitizenProfileRequest(BaseModel):
    Age: int = Field(..., ge=1, le=100, description="Citizen's age in years (1 to 100)")
    Gender: str = Field(..., description="Female or Male")
    Income: int = Field(..., ge=0, le=1000000, description="Annual household income in INR")
    Student: str = Field(default="No", description="Yes or No")
    GovtSchoolStudent: str = Field(default="No", description="Yes or No")
    Farmer: str = Field(default="No", description="Yes or No")
    Widow: str = Field(default="No", description="Yes or No")
    Disability: str = Field(default="No", description="Yes or No")
    WorkingWoman: str = Field(default="No", description="Yes or No")
    SC: str = Field(default="No", description="Yes or No")
    ST: str = Field(default="No", description="Yes or No")
    LandOwner: str = Field(default="No", description="Yes or No")
    Pregnant: str = Field(default="No", description="Yes or No")
    TNResident: str = Field(default="Yes", description="Yes or No")
    NoPermanentHouse: str = Field(default="No", description="Yes or No")
    BPL: str = Field(default="No", description="Yes or No")
    PrimaryEarnerDeceased: str = Field(default="No", description="Yes or No")
    GirlChild: str = Field(default="No", description="Yes or No")
    EnrolledTraining: str = Field(default="No", description="Yes or No")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 20,
                "Gender": "Female",
                "Income": 120000,
                "Student": "Yes",
                "GovtSchoolStudent": "Yes",
                "Farmer": "No",
                "Widow": "No",
                "Disability": "No",
                "WorkingWoman": "No",
                "SC": "No",
                "ST": "No",
                "LandOwner": "No",
                "Pregnant": "No",
                "TNResident": "Yes",
                "NoPermanentHouse": "No",
                "BPL": "No",
                "PrimaryEarnerDeceased": "No",
                "GirlChild": "No",
                "EnrolledTraining": "No"
            }
        }


class ShapFactor(BaseModel):
    label: str
    value: str
    impact_text: str
    score: float
    is_positive: bool


class SchemeResult(BaseModel):
    index: int
    name: str
    category: str
    department: str
    description: str
    is_eligible: bool
    confidence: float
    confidence_pct: float
    factors: List[ShapFactor]
    criteria: Dict[str, Any]


class PredictionSummary(BaseModel):
    total_evaluated: int
    eligible_count: int
    ineligible_count: int
    highest_confidence: float
    average_confidence: float


class PredictionResponse(BaseModel):
    status: str
    summary: PredictionSummary
    eligible_schemes: List[SchemeResult]
    ineligible_schemes: List[SchemeResult]
