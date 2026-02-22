"""Streamlit frontend for interactive manual testing / demo.

Provides a lightweight UI layer over the underlying classification
workflow so developers or stakeholders can drag & drop images and
inspect the minimal JSON output without needing the REST API.
"""

import streamlit as st
import base64
from PIL import Image
import io
from graph import ImageClassificationGraph
import config


def main():
    """Entrypoint for the Streamlit app.

    Handles: API key presence check, image upload, invoking the graph,
    and presenting the raw JSON output (mirroring the FastAPI contract).
    """
    
    st.set_page_config(
        page_title="Image Classification System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Environmental Image Classification System")
    st.markdown("""
    This system classifies images into three categories:
    - **Garbage**: Waste, litter, and pollution **in public spaces**
    - **Potholes**: Road damage and infrastructure issues  
    - **Deforestation**: Tree cutting and forest destruction
    
    **Output Format**: Raw JSON with four essential fields:
    - `category`: Classification result (garbage/potholes/deforestation/reject)
    - `severity`: Numeric score from 0-100 (null for rejected images)
    - `severity_level`: Descriptive severity level (null for rejected images)
    - `scale`: Scale information about the issue size/extent (null for rejected images)
    
    ⚠️ **Important**: Household/indoor garbage images will be automatically rejected. 
    Only public environmental issues are accepted for monitoring.
    
    All detailed analysis is processed in the backend for maximum precision and accuracy.
    """)
    
    # Check if API key is configured
    if not config.OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    # Lazy-init classification graph and keep in session for reuse.
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ImageClassificationGraph()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to classify"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            
            # Display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert image to base64 (mirrors FastAPI preprocessing)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        with col2:
            st.subheader("🤖 Classification Results")
            
            # Process button
            if st.button("🔍 Analyze & Classify", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Process the image
                        result = st.session_state.classifier.process_image(img_base64)
                        
                        # Display results
                        display_results(result)
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")


def display_results(result: dict):
    """Render the minimal response plus simple visual hints.

    Avoids exposing internal reasoning / confidence to keep parity with
    the public REST response shape.
    """
    
    # Display the raw JSON output with just the three essential fields
    st.subheader("📋 Classification Result (Raw JSON)")
    st.json(result)
    
    # Optional: Add a simple visual indicator for the category
    category = result.get('category', 'reject')
    severity = result.get('severity')
    severity_level = result.get('severity_level')
    scale = result.get('scale')
    
    if category == 'reject':
        st.error("❌ **Status: REJECTED**")
    else:
        category_emojis = {
            'garbage': '🗑️',
            'potholes': '🕳️',
            'deforestation': '🌳'
        }
        emoji = category_emojis.get(category, '❓')
        st.success(f"{emoji} **Classified as: {category.upper()}**")

        if severity is not None:
            st.info(f"📊 **Severity: {severity}/100** | **Level: {severity_level or 'N/A'}**")
            if scale:
                st.info(f"📏 **Scale: {scale}**")


if __name__ == "__main__":
    main()
