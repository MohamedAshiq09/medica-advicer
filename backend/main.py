from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Symptoms Checker API",
    description="AI-powered medical triage system for symptom assessment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomInput(BaseModel):
    text: str = Field(..., description="User's symptom description", min_length=1)
    age: Optional[int] = Field(None, description="User's age", ge=0, le=120)
    gender: Optional[str] = Field(None, description="User's gender (M/F/Other)")

class TriageResponse(BaseModel):
    triage_level: str
    confidence: float
    explanation: str
    detected_symptoms: List[str]
    probabilities: Dict[str, float]
    applied_overrides: List[Dict[str, Any]]
    applied_rules: List[Dict[str, Any]]
    follow_up_questions: List[str]
    timestamp: str
    model_info: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    llm_available: bool
    apriori_loaded: bool

class FeedbackInput(BaseModel):
    symptoms: List[str] = Field(..., description="List of detected symptoms")
    predicted_label: str = Field(..., description="What the model predicted")
    correct_label: str = Field(..., description="What the user says is correct")
    user_notes: Optional[str] = Field(None, description="Optional notes from user")
    severity_info: Optional[Dict[str, Any]] = Field(None, description="Severity information")

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_recorded: bool
    stats: Dict[str, Any]

agent_status = {
    "model_loaded": False,
    "apriori_loaded": False,
    "startup_time": None
}

def initialize_agents():
    logger.info("Starting Medical Symptoms Checker API...")
    
    try:
        # Try to load improved model first (with age/gender support)
        try:
            from src.improved_model_inference import load_model
            model_loaded = load_model()
            if model_loaded:
                logger.info("✅ Improved ML model (age/gender-aware) loaded successfully")
                agent_status["model_loaded"] = True
                agent_status["model_type"] = "improved_hybrid"
            else:
                logger.warning("Improved model not available, trying standard model...")
                raise Exception("Improved model not loaded")
        except:
            # Fallback to standard model
            from src.model_inference import load_model
            model_loaded = load_model()
            agent_status["model_loaded"] = model_loaded
            agent_status["model_type"] = "standard"
            if model_loaded:
                logger.info("Standard ML model loaded successfully")
            else:
                logger.warning("ML model not loaded - using fallback logic")
    except Exception as e:
        logger.error(f"Error loading ML model: {str(e)}")
        agent_status["model_loaded"] = False
        agent_status["model_type"] = "none"
    
    try:
        from src.rules_apriori import load_apriori_rules
        apriori_loaded = load_apriori_rules()
        agent_status["apriori_loaded"] = apriori_loaded
        if apriori_loaded:
            logger.info("Apriori rules loaded successfully")
        else:
            logger.info("Apriori rules not found - using default patterns")
    except Exception as e:
        logger.error(f"Error loading Apriori rules: {str(e)}")
        agent_status["apriori_loaded"] = False
    
    agent_status["startup_time"] = datetime.now().isoformat()
    logger.info("Medical Symptoms Checker API started successfully")

@app.on_event("startup")
async def startup_event():
    initialize_agents()

@app.get("/")
async def root():
    return {
        "message": "Medical Symptoms Checker API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        from src.llm_layer import is_llm_available
        llm_available = is_llm_available()
    except:
        llm_available = False
        
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=agent_status["model_loaded"],
        llm_available=llm_available,
        apriori_loaded=agent_status["apriori_loaded"]
    )

@app.post("/check_symptoms", response_model=TriageResponse)
async def check_symptoms(payload: SymptomInput):
    try:
        logger.info(f"Processing symptom check request: {payload.text[:50]}...")
        
        from src.preprocessing import preprocess_input
        from src.enhanced_triage import perform_triage
        from src.llm_layer import generate_follow_up_questions
        
        # Preprocess input
        features_dict, detected_symptoms = preprocess_input(
            payload.text, 
            payload.age, 
            payload.gender
        )
        
        logger.info(f"Detected symptoms: {detected_symptoms}")
        
        # Use enhanced triage engine with medical knowledge base
        triage_result = perform_triage(
            detected_symptoms,
            payload.text,
            payload.age,
            payload.gender
        )
        
        logger.info(f"Triage recommendation: {triage_result['triage_level']} (confidence: {triage_result['confidence']:.2f})")
        
        # Generate follow-up questions
        follow_up_questions = generate_follow_up_questions(detected_symptoms)
        
        # Calculate probabilities based on triage level and confidence
        confidence = triage_result['confidence']
        triage_level = triage_result['triage_level']
        
        # Distribute probabilities based on triage level
        if triage_level == "emergency":
            probabilities = {
                "emergency": confidence,
                "see_doctor": (1 - confidence) * 0.7,
                "self-care": (1 - confidence) * 0.3
            }
        elif triage_level == "see_doctor":
            probabilities = {
                "see_doctor": confidence,
                "emergency": (1 - confidence) * 0.4,
                "self-care": (1 - confidence) * 0.6
            }
        else:  # self-care
            probabilities = {
                "self-care": confidence,
                "see_doctor": (1 - confidence) * 0.7,
                "emergency": (1 - confidence) * 0.3
            }
        
        # Normalize probabilities to sum to 1.0
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}
        
        # Build response
        response = TriageResponse(
            triage_level=triage_result['triage_level'],
            confidence=triage_result['confidence'],
            explanation=triage_result['reason'],
            detected_symptoms=detected_symptoms,
            probabilities=probabilities,
            applied_overrides=[{
                "type": "medical_knowledge_base",
                "matched_condition": triage_result.get('matched_condition'),
                "severity": triage_result.get('severity'),
                "reason": triage_result['reason']
            }],
            applied_rules=[],
            follow_up_questions=follow_up_questions,
            timestamp=datetime.now().isoformat(),
            model_info={
                "engine": "enhanced_triage_with_medical_kb",
                "matched_condition": triage_result.get('matched_condition'),
                "severity": triage_result.get('severity'),
                "red_flags": triage_result.get('red_flags', []),
                "duration": triage_result.get('duration', 'Unknown')
            }
        )
        
        logger.info(f"Final recommendation: {triage_result['triage_level']} (confidence: {triage_result['confidence']:.2f})")
        return response
        
    except Exception as e:
        logger.error(f"Error processing symptom check: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/symptoms/vocabulary")
async def get_symptom_vocabulary():
    try:
        from src.config import SYMPTOM_VOCABULARY
        return {"symptom_vocabulary": SYMPTOM_VOCABULARY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status_endpoint():
    try:
        from src.model_inference import get_model_status
        from src.llm_layer import is_llm_available
        
        model_info = get_model_status()
        return {
            **model_info,
            "agent_status": agent_status,
            "llm_available": is_llm_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/preprocessing")
async def test_preprocessing(payload: SymptomInput):
    try:
        from src.preprocessing import preprocess_input
        
        features_dict, detected_symptoms = preprocess_input(
            payload.text, 
            payload.age, 
            payload.gender
        )
        return {
            "original_text": payload.text,
            "detected_symptoms": detected_symptoms,
            "features": features_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/record", response_model=FeedbackResponse)
async def record_feedback(payload: FeedbackInput):
    """
    Record user feedback on a prediction to improve the model
    """
    try:
        from src.feedback_learning import record_user_feedback, get_feedback_statistics
        
        success = record_user_feedback(
            symptoms=payload.symptoms,
            predicted_label=payload.predicted_label,
            user_correction=payload.correct_label,
            user_notes=payload.user_notes or "",
            severity_info=payload.severity_info
        )
        
        stats = get_feedback_statistics()
        
        logger.info(f"Feedback recorded: {payload.correct_label} (was_correct: {payload.predicted_label == payload.correct_label})")
        
        return FeedbackResponse(
            success=success,
            message="Feedback recorded successfully" if success else "Failed to record feedback",
            feedback_recorded=success,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get statistics about collected feedback
    """
    try:
        from src.feedback_learning import get_feedback_statistics
        
        stats = get_feedback_statistics()
        return {
            "feedback_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain")
async def trigger_retrain():
    """
    Trigger model retraining based on feedback
    """
    try:
        from src.feedback_learning import should_retrain_model, retrain_from_feedback
        
        if not should_retrain_model():
            return {
                "success": False,
                "message": "Retraining not needed yet",
                "reason": "Insufficient feedback or accuracy is acceptable"
            }
        
        logger.info("Starting model retraining from feedback...")
        result = retrain_from_feedback()
        
        if result.get("success"):
            logger.info(f"Model retrained successfully: {result}")
        else:
            logger.warning(f"Model retraining failed: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)