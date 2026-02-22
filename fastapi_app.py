from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
import base64
from io import BytesIO
from PIL import Image
from graph import ImageClassificationGraph
import config

# Initialize FastAPI app
# Create the FastAPI application with some descriptive metadata used by OpenAPI docs
app = FastAPI(
    title="Environmental Image Classification API",
    description="API for classifying images into garbage, potholes, deforestation, or reject categories",
    version="1.0.0"
)

# Add CORS middleware for React frontend
# Allow cross-origin requests from the frontend (React). In production restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace '*' with your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ImageRequest(BaseModel):
    """Pydantic model for requests that provide an image URL.

    Fields:
    - image_url: A validated URL pointing to an image resource.
    """
    image_url: HttpUrl
    
# Response model
class ClassificationResponse(BaseModel):
    """Pydantic model for the classification response returned by endpoints.

    Fields:
    - category: Predicted category (e.g. garbage, potholes, deforestation, reject)
    - severity: Numeric severity score (0-100) when applicable
    - severity_level: Human readable severity level (low/medium/high) when applicable
    - scale: Additional scale/context information (optional)
    """
    category: str
    severity: int | None
    severity_level: str | None
    scale: str | None

# Initialize the classification graph
# Instantiate the project's image classification graph / pipeline
classifier = ImageClassificationGraph()

def process_uploaded_file(file_content: bytes) -> str:
    """Convert raw uploaded image bytes into a normalized base64 JPEG string.

    Steps:
    1. Wrap raw bytes into a BytesIO so PIL can open it like a file.
    2. Use PIL to verify that the bytes represent a valid image.
    3. Re-open the image after verify() (verify() can close the file).
    4. Normalize mode: convert transparency-supporting images to RGB with white
       background, and convert any non-RGB images to RGB.
    5. Save the image into an in-memory buffer as JPEG and base64-encode it.

    Returns:
    - base64-encoded JPEG string suitable for downstream model input.
    """
    try:
        # Create BytesIO object from file content so PIL can read it
        image_bytes = BytesIO(file_content)

        # Open and verify it's a valid image. verify() checks file integrity
        # but may close the file, so we seek and re-open afterwards.
        try:
            image = Image.open(image_bytes)
            image.verify()  # quick integrity check

            # Reopen the image since verify() may close the file handle
            image_bytes.seek(0)
            image = Image.open(image_bytes)

        except Exception as e:
            # Raise a 400 so the client knows the upload was invalid
            raise HTTPException(status_code=400, detail=f"Cannot open image file: {str(e)}")

        # Handle images with alpha channels by compositing onto white background
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            # Convert paletted or grayscale images to RGB
            image = image.convert('RGB')

        # Save processed image to a buffer as JPEG and encode to base64
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    except HTTPException:
        # Re-raise HTTPExceptions unchanged
        raise
    except Exception as e:
        # Convert unexpected errors to client-friendly HTTP error
        raise HTTPException(status_code=400, detail=f"Unexpected error processing image: {str(e)}")

def download_and_encode_image(image_url: str) -> str:
    """Download an image from a public URL and convert it to a base64 JPEG string.

    This function performs defensive checks:
    - Ensures the HTTP response contains content
    - Attempts basic content-type checking but will still try to open bytes with PIL
      if the content-type header is missing or incorrect
    - Verifies image integrity with PIL.verify()
    - Normalizes image mode to RGB and encodes to base64 as JPEG
    """
    try:
        # Use a common browser User-Agent to reduce likelihood of blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Download the image bytes
        response = requests.get(image_url, timeout=30, headers=headers)
        response.raise_for_status()

        # Ensure we received bytes
        if not response.content:
            raise HTTPException(status_code=400, detail="Downloaded content is empty")

        # Basic content-type check; fall back to PIL detection if header is wrong
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
            # Try to detect if it's still an image despite wrong content-type header
            try:
                test_image = Image.open(BytesIO(response.content))
                test_image.verify()
            except Exception:
                raise HTTPException(status_code=400, detail=f"URL does not contain a valid image. Content-Type: {content_type}")

        # Open and verify the downloaded image using the same pipeline as uploads
        image_bytes = BytesIO(response.content)
        try:
            image = Image.open(image_bytes)
            image.verify()

            # Reopen after verify()
            image_bytes.seek(0)
            image = Image.open(image_bytes)

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot open image file: {str(e)}")

        # Normalize image mode (handle alpha channels and non-RGB modes)
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Encode to JPEG and return base64 string
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    except HTTPException:
        # Re-raise HTTPExceptions so caller can handle them directly
        raise
    except requests.RequestException as e:
        # Network-level errors converting to HTTP 400 for the client
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        # Fallback for unexpected errors
        raise HTTPException(status_code=400, detail=f"Unexpected error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic API info to help developers and the frontend.

    This returns a small JSON object describing available endpoints and expected
    fields so the API is discoverable from the browser or dev tools.
    """
    return {
        "message": "Environmental Image Classification API",
        "description": "Classify images using URL or direct upload",
        "endpoints": {
            "POST /classify": "Classify image from URL - send {\"image_url\": \"https://...\"}",
            "POST /classify-upload": "Classify uploaded image file - send multipart/form-data with 'file' field"
        },
        "categories": ["garbage", "potholes", "deforestation", "reject"],
        "severity_range": "0-100 (null for rejected images)",
        "fields": ["category", "severity", "severity_level", "scale"]
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(request: ImageRequest):
    """
    Classify an image from a URL
    
    - **image_url**: Direct URL to an image file (jpg, png, etc.)
    
    Returns classification with category, severity (0-100), severity_level, and scale
    """
    # Ensure the API key required by the classification pipeline is available
    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Download and encode the image to base64 for the classifier
        image_base64 = download_and_encode_image(str(request.image_url))

        # Pass the base64 image into the classification pipeline
        result = classifier.process_image(image_base64)

        # Return a validated response using the Pydantic model
        return ClassificationResponse(
            category=result["category"],
            severity=result["severity"],
            severity_level=result["severity_level"],
            scale=result["scale"]
        )

    except HTTPException:
        # Re-raise HTTPExceptions so clients receive the original error code
        raise
    except Exception as e:
        # Unexpected server error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/classify-upload", response_model=ClassificationResponse)
async def classify_uploaded_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image file
    
    - **file**: Image file (jpg, png, gif, webp, etc.)
    
    Returns classification with category, severity (0-100), severity_level, and scale
    """
    # Ensure required API key is present for classification
    if not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )

    # Basic validation: make sure the uploaded file is an image
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, gif, webp, etc.)"
        )

    try:
        # Read the uploaded file into memory (UploadFile provides async read())
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Convert uploaded bytes into normalized base64 JPEG string
        image_base64 = process_uploaded_file(file_content)

        # Run the classifier and return structured response
        result = classifier.process_image(image_base64)

        return ClassificationResponse(
            category=result["category"],
            severity=result["severity"],
            severity_level=result["severity_level"],
            scale=result["scale"]
        )

    except HTTPException:
        # Re-raise known HTTP errors
        raise
    except Exception as e:
        # Unexpected server error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check used by monitoring or the frontend.

    Returns the service status and whether the required API key is configured.
    """
    return {"status": "healthy", "api_key_configured": bool(config.OPENAI_API_KEY)}

if __name__ == "__main__":
    # Start the app with Uvicorn when running this file directly.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
