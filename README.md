# Environmental Image Classification System

This project classifies images into environmental categories: **Garbage**, **Potholes**, and **Deforestation**, with severity scoring from 0-100 and scale information about the size/extent of the issue. The system provides both a FastAPI REST API and a Streamlit web interface.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create environment file:**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

#### Option 1: FastAPI Server (Recommended for API usage)
```bash
python fastapi_app.py
```
- API will be available at: `http://localhost:8000`
- Interactive documentation: `http://localhost:8000/docs`
- Demo page: Open `demo.html` in your browser

#### Option 2: Streamlit Web Interface
```bash
streamlit run app.py
```
- Web interface will be available at: `http://localhost:8501`

## 📚 Documentation

### API Documentation
See **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** for comprehensive API usage guide including:
- Endpoint descriptions and examples
- React frontend integration code
- Error handling
- cURL examples

### Quick API Usage

**Classify from URL:**
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/image.jpg"}'
```

**Classify uploaded file:**
```bash
curl -X POST "http://localhost:8000/classify-upload" \
     -F "file=@/path/to/image.jpg"
```

## 🏗️ Project Structure

```
├── fastapi_app.py          # FastAPI REST API server
├── app.py                  # Streamlit web interface
├── config.py              # Configuration settings
├── demo.html              # HTML demo page for testing API
├── API_DOCUMENTATION.md   # Comprehensive API documentation
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models for structured data
├── nodes/
│   ├── __init__.py
│   ├── analysis_node.py   # Image analysis processing
│   └── classification_node.py  # Classification logic
└── graph/
    ├── __init__.py
    └── workflow.py         # LangGraph workflow orchestration
```

## 🎯 Features

- **Multiple Input Methods**: URL-based or file upload
- **Environmental Focus**: Specialized for garbage, potholes, and deforestation
- **Severity Scoring**: 0-100 scale with human-readable levels
- **Scale Information**: Contextual size/extent descriptions
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Web Interface**: User-friendly Streamlit interface
- **CORS Enabled**: Ready for frontend integration
- **Error Handling**: Comprehensive error responses

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| POST | `/classify` | Classify image from URL |
| POST | `/classify-upload` | Classify uploaded image file |
| GET | `/health` | Health check and API key status |
| GET | `/docs` | Interactive API documentation |

## 📊 Classification Categories

- **🗑️ Garbage**: Litter, waste, pollution
- **🕳️ Potholes**: Road damage, infrastructure issues  
- **🌳 Deforestation**: Tree cutting, forest clearing
- **❌ Reject**: Images that don't fit environmental categories

## 🌐 Frontend Integration

The API includes CORS support and is ready for integration with React, Vue, Angular, or any other frontend framework.
## 🧪 Testing

1. **Start the API server:**
   ```bash
   python fastapi_app.py
   ```

2. **Open the demo page:**
   Open `demo.html` in your web browser to test both URL and upload functionality.

3. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `OPENAI_MODEL` | OpenAI model to use (default: gpt-4o) | No |

## 📝 Response Format

```json
{
  "category": "garbage",
  "severity": 75,
  "severity_level": "moderate-high", 
  "scale": "large pile of litter in park area"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Streamlit and FastAPI interfaces
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
