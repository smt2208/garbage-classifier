from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.schemas import GraphState, ClassificationResult
import config


def classify_image_node(state: GraphState) -> GraphState:
    """Decision node: map prior analysis to category + severity.

    Applies rejection logic (e.g., indoor household garbage) early to avoid
    unnecessary model calls and to maintain deterministic handling of known
    non-environmental images.
    """
    try:
        if not state.analysis:
            state.error = "No analysis available for classification"
            return state

        # Short-circuit: indoor/household garbage is always rejected regardless
        # of other features to prevent false positives inflating metrics.
        if state.analysis.is_indoor_household:
            state.classification = ClassificationResult(
                category="reject",
                severity=None,
                severity_level=None,
                scale=None,
                confidence=0.95,  # High confidence due to explicit gating rule.
                reasoning="Image identified as household/indoor garbage which is not appropriate for environmental monitoring.",
            )
            return state

        # Structured output enforces schema compliance (severity bounds, etc.).
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.8,  # Slightly lower for consistency / reproducibility.
        ).with_structured_output(ClassificationResult)

        # Prompt includes explicit category taxonomy, rejection heuristics, and
        # severity calibration guidance to reduce ambiguity.
        classification_prompt = f"""
        Based on the detailed image analysis provided, classify this image into one of these categories:

        1. garbage: Litter, waste, dumping, public-space pollution (OUTDOOR ONLY)
        2. potholes: Road or pavement surface damage (holes, cracks, erosion)
        3. deforestation: Tree removal, stumps, cleared forest areas, logging
        4. reject: Anything else or indoor/household garbage / irrelevant content

        REJECTION RULES:
        - Indoor or household garbage → reject
        - Private personal trash not in public context → reject
        - Ambiguous content not clearly belonging to defined categories → reject

        SEVERITY GUIDELINES:
        - 0–30  : low / low-high (minor, limited impact)
        - 31–55 : moderate (noticeable, localized impact)
        - 56–75 : moderate-high (significant or growing issue)
        - 76–90 : high (large, impactful, requires action soon)
        - 91–100: extreme (severe, urgent intervention likely needed)

        Provide:
        - category (or reject)
        - severity (0–100) unless reject
        - severity_level (one of: low, low-high, moderate, moderate-high, high, extreme) unless reject
        - scale (concise phrase: e.g. "small pothole", "large garbage pile", "single tree", "extensive clearing") unless reject
        - confidence (0.0–1.0)
        - reasoning (succinct explanation referencing analysis evidence)

        IMAGE ANALYSIS INPUT:
        Description: {state.analysis.description}
        Objects: {', '.join(state.analysis.objects_detected)}
        Environment: {state.analysis.environment_type}
        Indoor/Household Flag: {state.analysis.is_indoor_household}
        Legitimacy Assessment: {state.analysis.legitimacy_assessment}
        Potential Issues: {', '.join(state.analysis.potential_issues)}

        Be conservative—only assign a non-reject category if clearly supported.
        """

        message = HumanMessage(content=classification_prompt)
        classification = llm.invoke([message])

        # Enforce invariant: reject must have null severity fields.
        if classification.category == "reject":
            classification.severity = None
            classification.severity_level = None
            classification.scale = None

        state.classification = classification
        return state

    except Exception as e:
        state.error = f"Error in image classification: {str(e)}"
        return state
