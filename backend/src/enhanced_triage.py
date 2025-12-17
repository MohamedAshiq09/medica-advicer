"""
Enhanced Triage Engine - Combines ML predictions with medical knowledge base
Provides real, evidence-based triage recommendations
"""
import logging
from typing import Dict, List, Tuple
from .medical_knowledge_base import (
    get_triage_recommendation,
    check_emergency,
    get_symptom_severity
)
from .feedback_learning import adjust_confidence

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
                      age: int = None,
                      gender: str = None) -> Dict:
        """
        Perform comprehensive triage using medical knowledge base
        
        Args:
            detected_symptoms: List of detected symptoms
            user_text: Original user input
            age: User age
            gender: User gender
            
        Returns:
            Dictionary with triage recommendation
        """
        
        if not detected_symptoms:
            return {
                "triage_level": "see_doctor",
                "confidence": 0.50,
                "reason": "No symptoms detected. Please describe your symptoms.",
                "advice": "Consult with a healthcare provider",
                "detected_symptoms": [],
                "matched_condition": None
            }
        
        logger.info(f"Performing triage for symptoms: {detected_symptoms}")
        
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
                "severity": "CRITICAL"
            }
        
        # Step 2: Get recommendation from medical knowledge base
        kb_recommendation = get_triage_recommendation(detected_symptoms)
        logger.info(f"Knowledge base recommendation: {kb_recommendation['triage_level']}")
        
        # Step 3: Calculate symptom severity
        avg_severity = self._calculate_average_severity(detected_symptoms, kb_recommendation.get('matched_condition'))
        logger.info(f"Average symptom severity: {avg_severity:.2f}/3")
        
        # Step 4: Adjust confidence based on feedback history
        adjusted_confidence = adjust_confidence(detected_symptoms, kb_recommendation['confidence'])
        logger.info(f"Adjusted confidence: {adjusted_confidence:.2f}")
        
        # Step 5: Apply age-based adjustments
        final_confidence = self._apply_age_adjustments(
            adjusted_confidence,
            age,
            detected_symptoms
        )
        
        # Step 6: Determine final triage level
        final_triage = self._determine_final_triage(
            kb_recommendation['triage_level'],
            final_confidence,
            avg_severity,
            detected_symptoms
        )
        
        # Step 7: Generate detailed reasoning
        reasoning = self._generate_reasoning(
            detected_symptoms,
            kb_recommendation,
            avg_severity,
            final_triage
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
            "kb_recommendation": kb_recommendation['triage_level']
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
    
    def _apply_age_adjustments(self, 
                              confidence: float,
                              age: int,
                              symptoms: List[str]) -> float:
        """
        Apply age-based adjustments to confidence
        
        Args:
            confidence: Base confidence
            age: User age
            symptoms: Detected symptoms
            
        Returns:
            Adjusted confidence
        """
        if age is None:
            return confidence
        
        # Children and elderly are higher risk
        if age < 5 or age > 65:
            # Increase confidence for conservative recommendations
            if any(s in symptoms for s in ['fever', 'vomiting', 'diarrhea']):
                confidence = min(0.95, confidence * 1.1)
        
        return confidence
    
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
                           final_triage: str) -> str:
        """Generate detailed reasoning for recommendation"""
        
        reasoning = f"Based on your symptoms ({', '.join(symptoms)}), "
        
        if kb_rec.get('matched_condition'):
            reasoning += f"this appears consistent with {kb_rec['matched_condition'].replace('_', ' ')}. "
        
        if severity >= 2.5:
            reasoning += "The severity of your symptoms requires immediate medical attention. "
        elif severity >= 2.0:
            reasoning += "Your symptoms warrant prompt medical evaluation. "
        else:
            reasoning += "Your symptoms may be manageable with home care, but monitor closely. "
        
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
