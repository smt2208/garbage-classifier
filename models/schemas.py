from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field


class ImageAnalysis(BaseModel):
    """Rich intermediate scene understanding.

    Produced by the analysis node. Intentionally verbose so downstream
    decisions (classification, severity) can rely on structured evidence
    instead of re-parsing natural language each step.
    """

    description: str = Field(
        description="Detailed description of what's visible in the image"
    )
    objects_detected: List[str] = Field(
        description="List of main objects, features, or elements detected in the image"
    )
    environment_type: str = Field(
        description="Type of environment (urban, rural, indoor, outdoor, etc.)"
    )
    is_indoor_household: bool = Field(
        description="True if this appears to be indoor/household garbage or personal living space, False for outdoor/public areas"
    )
    lighting_conditions: str = Field(
        description="Lighting conditions in the image (bright, dim, natural, artificial, etc.)"
    )
    image_quality: str = Field(
        description="Assessment of image quality (clear, blurry, high-resolution, etc.)"
    )
    potential_issues: List[str] = Field(
        description="Any potential environmental or infrastructure issues visible"
    )
    legitimacy_assessment: str = Field(
        description="Assessment of whether this represents a legitimate environmental/infrastructure concern vs household/personal waste"
    )


class ClassificationResult(BaseModel):
    """Compact decision output.

    Returned by the classification node. Only a subset of these fields are
    exposed publicly; internal fields (confidence, reasoning) support auditing
    and future refinement without expanding the external API surface.
    """

    category: Literal["garbage", "potholes", "deforestation", "reject"] = Field(
        description="Classification category or 'reject' if image doesn't fit any category"
    )
    severity: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Severity score from 0-100, or null if category is 'reject'"
    )
    severity_level: Optional[Literal["low", "low-high", "moderate", "moderate-high", "high", "extreme"]] = Field(
        default=None,
        description="Severity level category, or null if category is 'reject'"
    )
    scale: Optional[str] = Field(
        default=None,
        description="Scale information about the size/extent of the issue (e.g., 'small pothole', 'large garbage pile', 'single tree', 'forest area'), or null if category is 'reject'"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 to 1.0)"
    )
    reasoning: str = Field(
        description="Explanation for the classification and severity score"
    )


class GraphState(BaseModel):
    """Mutable state propagated through the workflow.

    Each node can enrich or annotate this structure. The final formatter node
    projects a subset to the API response while retaining full detail here.
    """

    image_data: str = Field(description="Base64 encoded image data")
    analysis: Optional[ImageAnalysis] = None
    classification: Optional[ClassificationResult] = None
    error: Optional[str] = None
    formatted_result: Optional[Dict[str, Any]] = None
