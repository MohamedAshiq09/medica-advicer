"""
Improved Model Inference - Age and Gender-Aware Predictions
Handles ML model loading and prediction with demographic features
"""
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
from .config import (
    TRIAGE_MODEL_PATH, 
    VECTORIZER_PATH, 
    TRIAGE_LEVELS,
    MIN_CONFIDENCE_THRESHOLD
)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedModelInferenceAgent:
    """
    Agent for age/gender-aware ML predictions
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Load all trained models and preprocessors"""
        try:
            if not TRIAGE_MODEL_PATH.exists():
                logger.warning(f"Model file not found: {TRIAGE_MODEL_PATH}")
                return False
            
            # Load main model
            self.model = joblib.load(TRIAGE_MODEL_PATH)
            logger.info("Triage model loaded successfully")
            
            # Load vectorizer
            if VECTORIZER_PATH.exists():
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("TF-IDF vectorizer loaded successfully")
            
            # Load scaler
            scaler_path = TRIAGE_MODEL_PATH.parent / "feature_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def create_hybrid_features(self, 
                              complaint_text: str,
                              age: Optional[int] = None,
                              gender: Optional[str] = None) -> np.ndarray:
        """
        Create hybrid feature vector combining text, age, and gender
        
        Args:
            complaint_text: User's symptom description
            age: User's age
            gender: User's gender (M/F)
            
        Returns:
            Feature array ready for model prediction
        """
        if not self.vectorizer:
            logger.error("Vectorizer not loaded")
            return None
        
        # Text features via TF-IDF
        text_vec = self.vectorizer.transform([complaint_text]).toarray()
        
        # Age and gender features (simple encoding)
        age = age or 40  # Default to adult age
        gender = (gender or 'M').upper()
        
        age_gender_features = np.array([[
            age,
            1 if gender == 'M' else 0  # is_male
        ]])
        
        # Combine all features
        X = np.hstack([text_vec, age_gender_features])
        
        # Scale only the age/gender features (text is already normalized by TF-IDF)
        if self.scaler:
            # Only scale the age/gender part
            age_gender_scaled = self.scaler.transform(age_gender_features)
            X = np.hstack([text_vec, age_gender_scaled])
        
        return X
    
    def predict_triage(self, 
                      complaint_text: str,
                      age: Optional[int] = None,
                      gender: Optional[str] = None) -> Tuple[str, Dict[str, float], float]:
        """
        Make age/gender-aware triage prediction
        
        Args:
            complaint_text: User's symptom description
            age: User's age
            gender: User's gender (M/F)
            
        Returns:
            Tuple of (predicted_label, probabilities_dict, confidence_score)
        """
        if not self.is_loaded:
            logger.warning("Models not loaded, attempting to load...")
            if not self.load_models():
                return "see_doctor", {"self-care": 0.0, "see_doctor": 1.0, "emergency": 0.0}, 0.5
        
        try:
            # Create hybrid features
            X = self.create_hybrid_features(complaint_text, age, gender)
            
            if X is None:
                return "see_doctor", {"self-care": 0.0, "see_doctor": 1.0, "emergency": 0.0}, 0.5
            
            # Get prediction
            predicted_class_idx = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            predicted_label = TRIAGE_LEVELS[predicted_class_idx]
            
            # Create probabilities dictionary
            prob_dict = {
                "self-care": float(probabilities[0]),
                "see_doctor": float(probabilities[1]), 
                "emergency": float(probabilities[2])
            }
            
            confidence = float(np.max(probabilities))
            
            # Age-based safety adjustments
            confidence = self._apply_age_safety_adjustments(
                confidence, predicted_label, age
            )
            
            # If confidence is too low, escalate to safety
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                if predicted_label == "self-care":
                    predicted_label = "see_doctor"
                logger.info(f"Low confidence ({confidence:.2f}), escalated to {predicted_label}")
            
            logger.info(f"Prediction: {predicted_label} (age: {age}, gender: {gender}, confidence: {confidence:.2f})")
            
            return predicted_label, prob_dict, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return "see_doctor", {"self-care": 0.0, "see_doctor": 1.0, "emergency": 0.0}, 0.5
    
    def _apply_age_safety_adjustments(self, 
                                     confidence: float,
                                     predicted_label: str,
                                     age: Optional[int]) -> float:
        """
        Apply age-based safety adjustments to confidence
        
        Args:
            confidence: Base confidence
            predicted_label: Predicted triage level
            age: User age
            
        Returns:
            Adjusted confidence
        """
        if age is None:
            return confidence
        
        # Very young children and elderly need conservative approach
        if age < 5:
            # Reduce confidence for self-care in very young children
            if predicted_label == "self-care":
                confidence = confidence * 0.7
        elif age >= 75:
            # Reduce confidence for self-care in very elderly
            if predicted_label == "self-care":
                confidence = confidence * 0.75
        elif age >= 65:
            # Slightly reduce confidence for self-care in elderly
            if predicted_label == "self-care":
                confidence = confidence * 0.85
        
        return confidence
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            "model_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "has_vectorizer": self.vectorizer is not None,
            "has_scaler": self.scaler is not None,
            "features": "hybrid (text + age + gender)"
        }
        return info

# Global instance
improved_model_inference_agent = ImprovedModelInferenceAgent()

def load_model() -> bool:
    """Load models at startup"""
    return improved_model_inference_agent.load_models()

def predict_triage(complaint_text: str,
                  age: Optional[int] = None,
                  gender: Optional[str] = None) -> Tuple[str, Dict[str, float], float]:
    """
    Main prediction function with age/gender awareness
    
    Args:
        complaint_text: User's symptom description
        age: User's age
        gender: User's gender (M/F)
        
    Returns:
        Tuple of (predicted_label, probabilities_dict, confidence_score)
    """
    return improved_model_inference_agent.predict_triage(complaint_text, age, gender)

def get_model_status() -> Dict:
    """Get current model status"""
    return improved_model_inference_agent.get_model_info()
