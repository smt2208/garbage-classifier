from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.schemas import GraphState, ImageAnalysis
import config


def analyze_image_node(state: GraphState) -> GraphState:
    """Vision + reasoning node producing a rich structured analysis.

    Produces an "ImageAnalysis" object that captures contextual signals used
    downstream for classification (environment type, objects, legitimacy hints,
    potential issues, etc.). This separation of concerns lets the classifier
    focus purely on decision + severity while this node extracts evidence.
    """
    try:
        # Instantiate model with structured output schema so we receive a
        # validated ImageAnalysis object instead of free-form JSON.
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.9,  # Slight creativity for richer scene description.
        ).with_structured_output(ImageAnalysis)

        # Prompt emphasizes: environmental context, indoor vs outdoor gating,
        # and scale/extent which influences severity.
        analysis_prompt = """
        Analyze this image in detail. Provide a comprehensive analysis including:
        1. Detailed description of visible elements
        2. All objects / features (concise list)
        3. Environment type (indoor household vs outdoor/public is critical)
        4. Lighting conditions
        5. Image quality assessment
        6. Environmental or infrastructure issues observable
        7. Scale/extent of the issue relative to its category (e.g. small pile vs large accumulation, small crack vs large pothole, single tree vs large clearing)

        KEY GATING RULE: Flag indoor/household garbage (kitchen waste, home bins, personal living spaces) distinctly — these should NOT be considered legitimate public environmental concerns.

        Provide only factual, observable details that can support downstream classification.
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state.image_data}"}},
            ]
        )

        analysis = llm.invoke([message])
        state.analysis = analysis
        return state

    except Exception as e:
        # Preserve error while allowing graph to continue to formatting.
        state.error = f"Error in image analysis: {str(e)}"
        return state
