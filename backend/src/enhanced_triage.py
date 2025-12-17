"""
Enhanced Triage Engine - Combines ML predictions with medical knowledge base
Provides real, evidence-based triage recommendations with age/gender awareness
"""
import logging
from typing import Dict, List, Tuple, Optional
from .medical_knowledge_base import (
    get_triage_recommendation,
    check_emergency,
    get_symptom_severity
)
from .feedback_learning import adjust_confidence
from .improved_model_inference import predict_triage as ml_predict_triage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTriageEngine:
    """
    Enhanced triage that combines ML with medical knowledge base
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            'self-care': 0.65,
            'see_doctor': 0.50,
            'emergency': 0.40
        }
    
    def perform_triage(self, 
                      detected_symptoms: List[str],
                      user_text: str = "",
                      age: Optional[int] = None,
                      gender: Optional[str] = None) -> Dict:
        """
        Perform comprehensive triage using ML + medical knowledge base
        Incorporates age and gender for accurate predictions
        
        Args:
            detected_symptoms: List of detected symptoms
            user_text: Original user input
            age: User age
            gender: User gender (M/F)
            
        Returns:
            Dictionary with triage recommendation
        """
        
        if not detected_symptoms and not user_text:
            return {
                "triage_level": "see_doctor",
                "confidence": 0.50,
                "reason": "No symptoms detected. Please describe your symptoms.",
                "advice": "Consult with a healthcare provider",
                "detected_symptoms": [],
                "matched_condition": None,
                "age": age,
                "gender": gender
            }
        
        logger.info(f"Performing triage - Age: {age}, Gender: {gender}, Symptoms: {detected_symptoms}")
        
        # Step 1: Check for emergency symptoms
        is_emergency, emergency_reason = check_emergency(detected_symptoms)
        if is_emergency:
            logger.warning(f"Emergency detected: {emergency_reason}")
            return {
                "triage_level": "emergency",
                "confidence": 0.95,
                "reason": emergency_reason,
                "advice": "SEEK IMMEDIATE MEDICAL ATTENTION. Call emergency services or go to nearest ER.",
                "detected_symptoms": detected_symptoms,
                "matched_condition": None,
                "severity": "CRITICAL",
                "age": age,
                "gender": gender
            }
        
        # Step 2: Get ML prediction with age/gender awareness
        ml_label, ml_probs, ml_confidence = ml_predict_triage(user_text, age, gender)
        logger.info(f"ML prediction: {ml_label} (confidence: {ml_confidence:.2f})")
        
        # Step 3: Get recommendation from medical knowledge base
        kb_recommendation = get_triage_recommendation(detected_symptoms)
        logger.info(f"Knowledge base recommendation: {kb_recommendation['triage_level']}")
        
        # Step 4: Calculate symptom severity
        avg_severity = self._calculate_average_severity(detected_symptoms, kb_recommendation.get('matched_condition'))
        logger.info(f"Average symptom severity: {avg_severity:.2f}/3")
        
        # Step 5: Combine ML and KB recommendations
        final_triage, final_confidence = self._combine_predictions(
            ml_label, ml_confidence, ml_probs,
            kb_recommendation['triage_level'], kb_recommendation['confidence'],
            avg_severity, age, detected_symptoms
        )
        
        # Step 6: Generate detailed reasoning
        reasoning = self._generate_reasoning(
            detected_symptoms,
            kb_recommendation,
            avg_severity,
            final_triage,
            age,
            gender
        )
        
        return {
            "triage_level": final_triage,
            "confidence": float(final_confidence),
            "reason": reasoning,
            "advice": kb_recommendation.get("advice", "Consult with healthcare provider"),
            "detected_symptoms": detected_symptoms,
            "matched_condition": kb_recommendation.get("matched_condition"),
            "severity": self._get_severity_label(avg_severity),
            "red_flags": kb_recommendation.get("red_flags", []),
            "duration": kb_recommendation.get("duration", "Unknown"),
            "kb_recommendation": kb_recommendation['triage_level'],
            "ml_recommendation": ml_label,
            "age": age,
            "gender": gender
        }
    
    def _calculate_average_severity(self, symptoms: List[str], 
                                   matched_condition: str = None) -> float:
        """Calculate average severity of symptoms"""
        if not symptoms:
            return 0
        
        # If we have a matched condition with severity_score, use it
        if matched_condition:
            from .medical_knowledge_base import knowledge_base
            condition_data = knowledge_base.get_condition_info(matched_condition)
            if 'severity_score' in condition_data:
                return condition_data['severity_score']
        
        total_severity = sum(get_symptom_severity(s) for s in symptoms)
        return total_severity / len(symptoms)
    
    def _combine_predictions(self,
                            ml_label: str,
                            ml_confidence: float,
                            ml_probs: Dict[str, float],
                            kb_label: str,
                            kb_confidence: float,
                            severity: float,
                            age: Optional[int],
                            symptoms: List[str]) -> Tuple[str, float]:
        """
        Combine ML and knowledge base predictions intelligently
        
        Args:
            ml_label: ML model prediction
            ml_confidence: ML model confidence
            ml_probs: ML model probabilities
            kb_label: Knowledge base prediction
            kb_confidence: Knowledge base confidence
            severity: Symptom severity
            age: User age
            symptoms: Detected symptoms
            
        Returns:
            Tuple of (final_label, final_confidence)
        """
        
        # If both agree, use that prediction with high confidence
        if ml_label == kb_label:
            combined_confidence = (ml_confidence + kb_confidence) / 2
            return ml_label, combined_confidence
        
        # If they disagree, escalate to safety
        # Emergency predictions always take priority
        if ml_label == "emergency" or kb_label == "emergency":
            return "emergency", max(ml_confidence, kb_confidence)
        
        # If one says see_doctor and other says self-care, choose see_doctor
        if (ml_label == "see_doctor" and kb_label == "self-care") or \
           (ml_label == "self-care" and kb_label == "see_doctor"):
            return "see_doctor", max(ml_confidence, kb_confidence) * 0.9
        
        # Age-based escalation
        if age is not None:
            if age < 5 or age >= 75:
                # Very young or very old: escalate self-care to see_doctor
                if ml_label == "self-care" or kb_label == "self-care":
                    return "see_doctor", max(ml_confidence, kb_confidence) * 0.85
        
        # High severity escalates
        if severity >= 2.5:
            if ml_label == "self-care" or kb_label == "self-care":
                return "see_doctor", max(ml_confidence, kb_confidence) * 0.9
        
        # Default to more conservative prediction
        return kb_label, kb_confidence
    
    def _determine_final_triage(self,
                               kb_triage: str,
                               confidence: float,
                               severity: float,
                               symptoms: List[str]) -> str:
        """
        Determine final triage level with safety checks
        
        Args:
            kb_triage: Knowledge base recommendation
            confidence: Adjusted confidence
            severity: Average symptom severity
            symptoms: Detected symptoms
            
        Returns:
            Final triage level
        """
        
        # Very high severity (3.0) always escalates to emergency
        if severity >= 3.0:
            return "emergency"
        
        # High severity (2.5+) escalates if not already emergency
        if severity >= 2.5:
            if kb_triage == "self-care":
                return "see_doctor"
            elif kb_triage == "see_doctor":
                return "emergency"
        
        # Medium-high severity (2.0+) escalates self-care to see_doctor
        if severity >= 2.0:
            if kb_triage == "self-care":
                return "see_doctor"
        
        # Low confidence escalates to safety
        if confidence < 0.5:
            if kb_triage == "self-care":
                return "see_doctor"
        
        # High-risk symptoms never allow self-care
        high_risk = {'chest_pain', 'shortness_breath', 'severe_headache', 'stiff_neck'}
        if any(s in high_risk for s in symptoms):
            if kb_triage == "self-care":
                return "see_doctor"
        
        return kb_triage
    
    def _generate_reasoning(self,
                           symptoms: List[str],
                           kb_rec: Dict,
                           severity: float,
                           final_triage: str,
                           age: Optional[int] = None,
                           gender: Optional[str] = None) -> str:
        """Generate detailed reasoning for recommendation"""
        
        reasoning = f"Based on your symptoms ({', '.join(symptoms)}), "
        
        # Add demographic context
        if age is not None:
            if age < 12:
                reasoning += f"as a child (age {age}), "
            elif age >= 65:
                reasoning += f"as a senior (age {age}), "
        
        if kb_rec.get('matched_condition'):
            reasoning += f"this appears consistent with {kb_rec['matched_condition'].replace('_', ' ')}. "
        
        if severity >= 2.5:
            reasoning += "The severity of your symptoms requires immediate medical attention. "
        elif severity >= 2.0:
            reasoning += "Your symptoms warrant prompt medical evaluation. "
        else:
            reasoning += "Your symptoms may be manageable with home care, but monitor closely. "
        
        # Age-specific advice
        if age is not None and age < 5:
            reasoning += "Given your child's young age, medical evaluation is recommended. "
        elif age is not None and age >= 75:
            reasoning += "Given your age, medical evaluation is recommended. "
        
        if final_triage == "emergency":
            reasoning += "SEEK IMMEDIATE MEDICAL ATTENTION."
        elif final_triage == "see_doctor":
            reasoning += "Please see a healthcare provider for proper diagnosis and treatment."
        else:
            reasoning += "Monitor your symptoms and seek care if they worsen."
        
        return reasoning
    
    def _get_severity_label(self, severity: float) -> str:
        """Get human-readable severity label"""
        if severity >= 2.5:
            return "CRITICAL"
        elif severity >= 2.0:
            return "HIGH"
        elif severity >= 1.5:
            return "MODERATE"
        else:
            return "LOW"

# Global instance
triage_engine = EnhancedTriageEngine()

def perform_triage(detected_symptoms: List[str],
                  user_text: str = "",
                  age: int = None,
                  gender: str = None) -> Dict:
    """
    Main function for enhanced triage
    
    Args:
        detected_symptoms: List of detected symptoms
        user_text: Original user input
        age: User age
        gender: User gender
        
    Returns:
        Triage recommendation
    """
    return triage_engine.perform_triage(detected_symptoms, user_text, age, gender)
