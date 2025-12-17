"""
Improved Inference Pipeline - Combines model predictions with feedback learning
Provides more nuanced predictions with confidence adjustments
"""
import logging
from typing import Dict, List, Tuple, Optional
from .model_inference import predict_triage as base_predict_triage
from .feedback_learning import (
    adjust_confidence, 
    should_retrain_model,
    retrain_from_feedback,
    get_feedback_statistics
)
from .config import TRIAGE_LEVELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedInferenceEngine:
    """
    Enhanced inference that combines ML predictions with feedback learning
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            'self-care': 0.65,      # Need higher confidence for self-care
            'see_doctor': 0.50,     # Medium confidence for doctor visit
            'emergency': 0.40       # Lower threshold for emergency (safety first)
        }
        
        # Symptom severity multipliers
        self.severity_multipliers = {
            'fever': 1.0,
            'headache': 1.0,
            'cough': 0.8,
            'chest_pain': 2.0,      # High risk
            'shortness_breath': 2.0, # High risk
            'severe_headache': 2.5,
            'dizziness': 1.2,
            'vomiting': 1.3,
            'abdominal_pain': 1.2,
            'fatigue': 0.7
        }
    
    def predict_with_feedback(self, 
                             features_dict: Dict,
                             detected_symptoms: List[str]) -> Dict:
        """
        Make prediction with feedback-based adjustments
        
        Args:
            features_dict: Feature dictionary from preprocessing
            detected_symptoms: List of detected symptom keys
            
        Returns:
            Dictionary with prediction, confidence, and reasoning
        """
        # Get base prediction from model
        base_label, base_probs, base_confidence = base_predict_triage(features_dict)
        
        # Adjust confidence based on feedback history
        adjusted_confidence = adjust_confidence(detected_symptoms, base_confidence)
        
        # Apply severity-based adjustments
        severity_adjustment = self._calculate_severity_adjustment(detected_symptoms)
        
        # Combine adjustments
        final_confidence = min(0.95, adjusted_confidence * severity_adjustment)
        
        # Determine final prediction
        final_label = self._determine_final_label(
            base_label, 
            base_probs, 
            final_confidence,
            detected_symptoms
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            detected_symptoms,
            base_label,
            final_label,
            final_confidence,
            severity_adjustment
        )
        
        # Check if retraining is needed
        needs_retrain = should_retrain_model()
        
        return {
            "prediction": final_label,
            "confidence": float(final_confidence),
            "base_prediction": base_label,
            "base_confidence": float(base_confidence),
            "adjusted_confidence": float(adjusted_confidence),
            "probabilities": base_probs,
            "detected_symptoms": detected_symptoms,
            "severity_adjustment": float(severity_adjustment),
            "reasoning": reasoning,
            "needs_retrain": needs_retrain,
            "feedback_stats": get_feedback_statistics()
        }
    
    def _calculate_severity_adjustment(self, symptoms: List[str]) -> float:
        """
        Calculate severity adjustment multiplier based on symptoms
        
        Args:
            symptoms: List of detected symptoms
            
        Returns:
            Adjustment multiplier (0.5 to 2.0)
        """
        if not symptoms:
            return 1.0
        
        total_multiplier = 0
        for symptom in symptoms:
            total_multiplier += self.severity_multipliers.get(symptom, 1.0)
        
        avg_multiplier = total_multiplier / len(symptoms)
        
        # Clamp between 0.5 and 2.0
        return max(0.5, min(2.0, avg_multiplier))
    
    def _determine_final_label(self,
                              base_label: str,
                              base_probs: Dict[str, float],
                              confidence: float,
                              symptoms: List[str]) -> str:
        """
        Determine final prediction label with nuanced logic
        
        Args:
            base_label: Base model prediction
            base_probs: Probability dictionary
            confidence: Adjusted confidence
            symptoms: Detected symptoms
            
        Returns:
            Final prediction label
        """
        # High-risk symptoms always escalate
        high_risk = {'chest_pain', 'shortness_breath', 'severe_headache'}
        if any(s in high_risk for s in symptoms):
            if base_label == 'self-care':
                return 'see_doctor'
            elif base_label == 'see_doctor' and confidence > 0.7:
                return 'emergency'
        
        # If confidence is too low, escalate to safety
        if confidence < 0.5:
            return 'see_doctor'
        
        # If probabilities are close, be conservative
        probs_list = sorted(base_probs.values(), reverse=True)
        if len(probs_list) >= 2:
            prob_diff = probs_list[0] - probs_list[1]
            if prob_diff < 0.15:  # Close call
                if base_label == 'self-care':
                    return 'see_doctor'
        
        return base_label
    
    def _generate_reasoning(self,
                           symptoms: List[str],
                           base_label: str,
                           final_label: str,
                           confidence: float,
                           severity_adj: float) -> str:
        """
        Generate human-readable reasoning for the prediction
        
        Args:
            symptoms: Detected symptoms
            base_label: Base prediction
            final_label: Final prediction
            confidence: Final confidence
            severity_adj: Severity adjustment applied
            
        Returns:
            Reasoning string
        """
        reasoning = f"Detected symptoms: {', '.join(symptoms)}. "
        
        if base_label != final_label:
            reasoning += f"Initial prediction was '{base_label}', "
            reasoning += f"upgraded to '{final_label}' due to "
            
            if severity_adj > 1.2:
                reasoning += "symptom severity. "
            else:
                reasoning += "safety considerations. "
        
        if confidence < 0.6:
            reasoning += "Confidence is moderate - consider consulting a healthcare provider. "
        elif confidence > 0.8:
            reasoning += "High confidence in this assessment. "
        
        return reasoning

# Global instance
inference_engine = ImprovedInferenceEngine()

def predict_with_feedback(features_dict: Dict, 
                         detected_symptoms: List[str]) -> Dict:
    """
    Main function for improved prediction with feedback
    
    Args:
        features_dict: Feature dictionary from preprocessing
        detected_symptoms: List of detected symptoms
        
    Returns:
        Dictionary with detailed prediction information
    """
    return inference_engine.predict_with_feedback(features_dict, detected_symptoms)

def trigger_model_retrain() -> Dict:
    """
    Trigger model retraining if needed
    
    Returns:
        Retraining results
    """
    if should_retrain_model():
        logger.info("Triggering model retrain based on feedback")
        return retrain_from_feedback()
    else:
        return {"success": False, "reason": "Retraining not needed yet"}
