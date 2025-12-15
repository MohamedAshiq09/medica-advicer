"""
Model Inference Agent - Handles ML model loading and prediction
Responsibility: Load trained models and provide triage predictions
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from .config import (
    TRIAGE_MODEL_PATH, 
    VECTORIZER_PATH, 
    LABEL_ENCODER_PATH,
    TRIAGE_LEVELS,
    MIN_CONFIDENCE_THRESHOLD
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInferenceAgent:
    """
    Agent responsible for loading ML models and making triage predictions
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_columns = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """
        Load all trained models and preprocessors
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Check if model files exist
            if not TRIAGE_MODEL_PATH.exists():
                logger.warning(f"Model file not found: {TRIAGE_MODEL_PATH}")
                return False
                
            # Load main triage model
            self.model = joblib.load(TRIAGE_MODEL_PATH)
            logger.info("Triage model loaded successfully")
            
            # Load vectorizer if it exists (for text-based models)
            if VECTORIZER_PATH.exists():
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("TF-IDF vectorizer loaded successfully")
            
            # Load label encoder if it exists
            if LABEL_ENCODER_PATH.exists():
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                logger.info("Label encoder loaded successfully")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def create_model_features(self, features_dict: Dict) -> np.ndarray:
        """
        Convert feature dictionary to model input format
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Feature array ready for model prediction
        """
        # Define expected feature columns for the model
        symptom_features = [
            'fever', 'headache', 'cough', 'chest_pain', 'shortness_breath',
            'nausea', 'vomiting', 'diarrhea', 'fatigue', 'dizziness',
            'sore_throat', 'runny_nose', 'muscle_pain', 'abdominal_pain'
        ]
        
        severity_features = ['pain_severity', 'duration_severity', 'intensity_severity']
        
        demographic_features = ['age_group', 'gender']
        
        other_features = ['symptom_count', 'text_length']
        
        # Create feature vector
        feature_vector = []
        
        # Add symptom flags (binary)
        for symptom in symptom_features:
            feature_vector.append(1 if features_dict.get(symptom, False) else 0)
        
        # Add severity scores (0-3)
        for severity in severity_features:
            feature_vector.append(features_dict.get(severity, 0))
        
        # Add demographic features (encoded)
        age_encoding = {'child': 0, 'adult': 1, 'senior': 2, 'unknown': 1}
        gender_encoding = {'male': 0, 'female': 1, 'm': 0, 'f': 1, 'unknown': 0}
        
        feature_vector.append(age_encoding.get(features_dict.get('age_group', 'unknown'), 1))
        feature_vector.append(gender_encoding.get(features_dict.get('gender', 'unknown'), 0))
        
        # Add other numerical features
        feature_vector.append(features_dict.get('symptom_count', 0))
        feature_vector.append(min(features_dict.get('text_length', 0), 100))  # Cap at 100 words
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_triage(self, features_dict: Dict) -> Tuple[str, Dict[str, float], float]:
        """
        Make triage prediction using loaded model
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Tuple of (predicted_label, probabilities_dict, confidence_score)
        """
        if not self.is_loaded:
            logger.warning("Models not loaded, attempting to load...")
            if not self.load_models():
                return "see_doctor", {"self-care": 0.0, "see_doctor": 1.0, "emergency": 0.0}, 0.5
        
        try:
            # Handle text-based model (TF-IDF + classifier)
            if self.vectorizer is not None:
                text_features = self.vectorizer.transform([features_dict.get('cleaned_text', '')])
                probabilities = self.model.predict_proba(text_features)[0]
            else:
                # Handle feature-based model
                model_features = self.create_model_features(features_dict)
                probabilities = self.model.predict_proba(model_features)[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_label = TRIAGE_LEVELS[predicted_class_idx]
            
            # Create probabilities dictionary
            prob_dict = {
                "self-care": float(probabilities[0]),
                "see_doctor": float(probabilities[1]), 
                "emergency": float(probabilities[2])
            }
            
            # Calculate confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # If confidence is too low, default to "see_doctor"
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                predicted_label = "see_doctor"
                logger.info(f"Low confidence ({confidence:.2f}), defaulting to see_doctor")
            
            return predicted_label, prob_dict, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Safe fallback
            return "see_doctor", {"self-care": 0.0, "see_doctor": 1.0, "emergency": 0.0}, 0.5
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "has_vectorizer": self.vectorizer is not None,
            "has_label_encoder": self.label_encoder is not None
        }
        
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importance"] = True
        
        return info

# Global instance
model_inference_agent = ModelInferenceAgent()

def load_model() -> bool:
    """
    Load models at startup
    
    Returns:
        True if successful, False otherwise
    """
    return model_inference_agent.load_models()

def predict_triage(features_dict: Dict) -> Tuple[str, Dict[str, float], float]:
    """
    Main prediction function to be called by API
    
    Args:
        features_dict: Dictionary of extracted features
        
    Returns:
        Tuple of (predicted_label, probabilities_dict, confidence_score)
    """
    return model_inference_agent.predict_triage(features_dict)

def get_model_status() -> Dict:
    """
    Get current model status
    
    Returns:
        Dictionary with model status information
    """
    return model_inference_agent.get_model_info()