"""
Simple Medical Symptoms Checker API - Test Version
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Symptoms Checker API",
    description="AI-powered medical triage system for symptom assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomInput(BaseModel):
    text: str
    age: Optional[int] = None
    gender: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Medical Symptoms Checker API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is working"}

@app.post("/test")
async def test_endpoint(payload: SymptomInput):
    """Simple test endpoint"""
    try:
        # Import here to avoid startup issues
        from src.preprocessing import preprocess_input
        
        features_dict, detected_symptoms = preprocess_input(
            payload.text, 
            payload.age, 
            payload.gender
        )
        
        return {
            "input_text": payload.text,
            "detected_symptoms": detected_symptoms,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)