from typing import Dict, Any
from langgraph.graph import StateGraph, END
from models.schemas import GraphState
from nodes import analyze_image_node, classify_image_node


class ImageClassificationGraph:
    """Encapsulates the LangGraph workflow used for image classification.

    The workflow proceeds through three conceptual stages:
    1. "analyze"  - detailed semantic / contextual image analysis
    2. "classify" - category + severity assignment informed by analysis
    3. "format_output" - reduce rich internal state to minimal API payload

    Keeping this orchestration isolated allows alternative frontends (FastAPI,
    Streamlit, batch jobs) to reuse identical classification logic.
    """

    def __init__(self):
        # Pre-compile the graph once per process for efficiency.
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct and compile the LangGraph workflow definition."""

        workflow = StateGraph(GraphState)

        # Register functional nodes (pure functions operating on GraphState).
        workflow.add_node("analyze", analyze_image_node)
        workflow.add_node("classify", classify_image_node)
        workflow.add_node("format_output", self._format_output_node)

        # Linear flow: analyze -> classify -> format_output -> END
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "classify")
        workflow.add_edge("classify", "format_output")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def _format_output_node(self, state: GraphState) -> GraphState:
        """Reduce internal rich state to the public minimal response.

        The public contract deliberately exposes only four fields to keep the
        external API stable and concise while allowing internal evolution
        (e.g., richer reasoning, confidence calibration) without breaking clients.
        """
        try:
            if state.error:
                # Error path: always normalize to a safe reject payload.
                result = {
                    "category": "reject",
                    "severity": None,
                    "severity_level": None,
                    "scale": None,
                }
            elif state.classification:
                # Happy path: project only essential fields.
                result = {
                    "category": state.classification.category,
                    "severity": state.classification.severity,
                    "severity_level": state.classification.severity_level,
                    "scale": state.classification.scale,
                }
            else:
                # Defensive fallback when classification missing unexpectedly.
                result = {
                    "category": "reject",
                    "severity": None,
                    "severity_level": None,
                    "scale": None,
                }

            state.formatted_result = result
            # Rich data (analysis, reasoning, confidence) is preserved on the
            # state object for internal logging / future auditing.
            return state

        except Exception as e:
            # Ensure downstream always receives a valid shaped state.
            state.error = f"Error formatting output: {str(e)}"
            state.formatted_result = {
                "category": "reject",
                "severity": None,
                "severity_level": None,
                "scale": None,
            }
            return state

    def process_image(self, image_base64: str) -> Dict[str, Any]:
        """Public API to run an image through the compiled workflow.

        Args:
            image_base64: Base64-encoded JPEG (or convertible) image data

        Returns:
            Dict with minimal fields required by external consumers.
        """
        initial_state = GraphState(image_data=image_base64)
        final_state = self.graph.invoke(initial_state)

        # Extract formatted result with defensive fallbacks.
        formatted_result = None
        try:
            if hasattr(final_state, "formatted_result"):
                formatted_result = final_state.formatted_result
            elif isinstance(final_state, dict):  # Edge: alternate return style
                formatted_result = final_state.get("formatted_result")
        except Exception:
            formatted_result = None

        if formatted_result is not None:
            return formatted_result

        # Absolute fallback to safe reject payload.
        return {
            "category": "reject",
            "severity": None,
            "severity_level": None,
            "scale": None,
        }
