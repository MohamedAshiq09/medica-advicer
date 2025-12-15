"""
Safety Layer Agent - Handles critical safety overrides and medical disclaimers
Responsibility: Apply hard-coded safety rules and ensure appropriate medical disclaimers
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from .config import RED_FLAG_SYMPTOMS, MEDICAL_DISCLAIMER, TRIAGE_DESCRIPTIONS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLayerAgent:
    """
    Agent responsible for applying safety overrides and medical disclaimers
    """
    
    def __init__(self):
        self.red_flag_patterns = self._initialize_red_flag_patterns()
        self.emergency_combinations = self._initialize_emergency_combinations()
        self.safety_overrides_applied = []
        
    def _initialize_red_flag_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize red flag symptom patterns that require immediate attention
        
        Returns:
            Dictionary of red flag categories and their patterns
        """
        return {
            "cardiac_emergency": [
                "severe chest pain",
                "crushing chest pain", 
                "chest pain radiating",
                "chest pain with shortness of breath",
                "heart attack symptoms"
            ],
            
            "respiratory_emergency": [
                "severe difficulty breathing",
                "cannot breathe",
                "gasping for air",
                "blue lips",
                "blue fingernails"
            ],
            
            "neurological_emergency": [
                "sudden severe headache",
                "worst headache of life",
                "loss of consciousness",
                "fainting",
                "seizure",
                "stroke symptoms",
                "sudden confusion",
                "sudden vision loss",
                "sudden speech problems"
            ],
            
            "severe_bleeding": [
                "severe bleeding",
                "heavy bleeding",
                "bleeding that won't stop",
                "internal bleeding"
            ],
            
            "allergic_emergency": [
                "severe allergic reaction",
                "anaphylaxis",
                "swelling of face",
                "swelling of throat",
                "difficulty swallowing"
            ],
            
            "trauma_emergency": [
                "severe injury",
                "broken bone",
                "head injury",
                "severe burn"
            ]
        }
    
    def _initialize_emergency_combinations(self) -> List[Dict]:
        """
        Initialize symptom combinations that indicate emergency
        
        Returns:
            List of emergency symptom combinations
        """
        return [
            {
                "symptoms": ["chest_pain", "shortness_breath"],
                "description": "Chest pain with breathing difficulty",
                "priority": "immediate"
            },
            {
                "symptoms": ["chest_pain", "dizziness"],
                "description": "Chest pain with dizziness",
                "priority": "immediate"
            },
            {
                "symptoms": ["severe_headache", "fever", "vomiting"],
                "description": "Severe headache with fever and vomiting",
                "priority": "immediate"
            },
            {
                "symptoms": ["abdominal_pain", "vomiting", "fever"],
                "description": "Severe abdominal symptoms",
                "priority": "urgent"
            },
            {
                "symptoms": ["shortness_breath", "chest_pain", "fatigue"],
                "description": "Multiple cardiac symptoms",
                "priority": "immediate"
            }
        ]
    
    def check_red_flag_symptoms(self, text: str, detected_symptoms: List[str]) -> Tuple[bool, List[str]]:
        """
        Check for red flag symptoms in text and detected symptoms
        
        Args:
            text: Original user input text
            detected_symptoms: List of detected symptom keys
            
        Returns:
            Tuple of (has_red_flags, list_of_red_flags_found)
        """
        text_lower = text.lower()
        red_flags_found = []
        
        # Check text for red flag patterns
        for category, patterns in self.red_flag_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    red_flags_found.append({
                        "category": category,
                        "pattern": pattern,
                        "source": "text_pattern"
                    })
        
        # Check for emergency symptom combinations
        detected_set = set(detected_symptoms)
        for combo in self.emergency_combinations:
            combo_symptoms = set(combo["symptoms"])
            if combo_symptoms.issubset(detected_set):
                red_flags_found.append({
                    "category": "symptom_combination",
                    "pattern": combo["description"],
                    "symptoms": combo["symptoms"],
                    "priority": combo["priority"],
                    "source": "symptom_combination"
                })
        
        # Check for specific high-risk individual symptoms
        high_risk_symptoms = {
            "chest_pain": "Chest pain requires medical evaluation",
            "shortness_breath": "Breathing difficulties need immediate attention",
            "severe_headache": "Severe headache may indicate serious condition"
        }
        
        for symptom in detected_symptoms:
            if symptom in high_risk_symptoms:
                # Check if it's described as severe in the text
                if any(severity in text_lower for severity in ["severe", "intense", "unbearable", "worst"]):
                    red_flags_found.append({
                        "category": "high_risk_symptom",
                        "pattern": high_risk_symptoms[symptom],
                        "symptom": symptom,
                        "source": "individual_symptom"
                    })
        
        return len(red_flags_found) > 0, red_flags_found
    
    def apply_safety_overrides(self, text: str, 
                             detected_symptoms: List[str],
                             base_prediction: str, 
                             base_confidence: float) -> Tuple[str, float, List[Dict]]:
        """
        Apply safety overrides based on red flag symptoms
        
        Args:
            text: Original user input text
            detected_symptoms: List of detected symptom keys
            base_prediction: Original prediction
            base_confidence: Original confidence
            
        Returns:
            Tuple of (final_prediction, final_confidence, applied_overrides)
        """
        has_red_flags, red_flags = self.check_red_flag_symptoms(text, detected_symptoms)
        applied_overrides = []
        
        final_prediction = base_prediction
        final_confidence = base_confidence
        
        if has_red_flags:
            # Determine override level based on red flags
            immediate_priority = any(
                flag.get("priority") == "immediate" or 
                flag.get("category") in ["cardiac_emergency", "respiratory_emergency", "neurological_emergency"]
                for flag in red_flags
            )
            
            if immediate_priority:
                final_prediction = "emergency"
                final_confidence = 0.95
                applied_overrides.append({
                    "type": "emergency_override",
                    "reason": "Critical symptoms detected requiring immediate attention",
                    "red_flags": red_flags
                })
                logger.warning(f"Emergency override applied due to: {[flag['pattern'] for flag in red_flags]}")
            
            elif base_prediction == "self-care":
                # Never allow self-care when red flags are present
                final_prediction = "see_doctor"
                final_confidence = max(0.8, base_confidence)
                applied_overrides.append({
                    "type": "safety_upgrade",
                    "reason": "Upgraded from self-care due to concerning symptoms",
                    "red_flags": red_flags
                })
                logger.info(f"Safety upgrade applied due to: {[flag['pattern'] for flag in red_flags]}")
        
        # Additional safety checks
        symptom_count = len(detected_symptoms)
        if symptom_count >= 5 and base_prediction == "self-care":
            final_prediction = "see_doctor"
            final_confidence = max(0.75, base_confidence)
            applied_overrides.append({
                "type": "multiple_symptoms_override",
                "reason": f"Multiple symptoms ({symptom_count}) warrant medical evaluation",
                "symptom_count": symptom_count
            })
        
        self.safety_overrides_applied = applied_overrides
        return final_prediction, final_confidence, applied_overrides
    
    def generate_safety_message(self, prediction: str, 
                               applied_overrides: List[Dict],
                               detected_symptoms: List[str]) -> str:
        """
        Generate appropriate safety message based on prediction and overrides
        
        Args:
            prediction: Final triage prediction
            applied_overrides: List of applied safety overrides
            detected_symptoms: List of detected symptoms
            
        Returns:
            Safety message string
        """
        base_message = TRIAGE_DESCRIPTIONS.get(prediction, "Consult with a healthcare professional")
        
        # Add specific warnings based on overrides
        warnings = []
        
        for override in applied_overrides:
            if override["type"] == "emergency_override":
                warnings.append("⚠️ URGENT: Your symptoms may indicate a medical emergency. Seek immediate medical attention or call emergency services.")
            elif override["type"] == "safety_upgrade":
                warnings.append("⚠️ CAUTION: Your symptoms require medical evaluation. Do not delay seeking care.")
            elif override["type"] == "multiple_symptoms_override":
                warnings.append("⚠️ NOTICE: Multiple symptoms present. Medical evaluation recommended.")
        
        # Add symptom-specific warnings
        if "chest_pain" in detected_symptoms:
            warnings.append("• Chest pain should always be evaluated by a medical professional")
        
        if "shortness_breath" in detected_symptoms:
            warnings.append("• Breathing difficulties require prompt medical attention")
        
        # Combine messages
        full_message = base_message
        if warnings:
            full_message += "\n\n" + "\n".join(warnings)
        
        # Always add medical disclaimer
        full_message += "\n\n" + MEDICAL_DISCLAIMER
        
        return full_message
    
    def validate_prediction_safety(self, prediction: str, 
                                 confidence: float,
                                 detected_symptoms: List[str]) -> Dict[str, any]:
        """
        Validate that the prediction is safe given the symptoms
        
        Args:
            prediction: Triage prediction
            confidence: Prediction confidence
            detected_symptoms: List of detected symptoms
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_safe": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check for unsafe combinations
        high_risk_symptoms = ["chest_pain", "shortness_breath", "severe_headache"]
        present_high_risk = [s for s in detected_symptoms if s in high_risk_symptoms]
        
        if present_high_risk and prediction == "self-care":
            validation_result["is_safe"] = False
            validation_result["warnings"].append(
                f"Self-care not recommended with symptoms: {', '.join(present_high_risk)}"
            )
            validation_result["recommendations"].append("Upgrade to 'see_doctor' or 'emergency'")
        
        # Check confidence levels
        if confidence < 0.6 and prediction == "emergency":
            validation_result["warnings"].append("Low confidence for emergency prediction")
            validation_result["recommendations"].append("Consider additional safety checks")
        
        return validation_result

# Global instance
safety_agent = SafetyLayerAgent()

def apply_safety_overrides(text: str,
                          detected_symptoms: List[str], 
                          base_prediction: str,
                          base_confidence: float) -> Tuple[str, float, List[Dict]]:
    """
    Main function to apply safety overrides
    
    Args:
        text: Original user input text
        detected_symptoms: List of detected symptom keys
        base_prediction: Original prediction
        base_confidence: Original confidence
        
    Returns:
        Tuple of (final_prediction, final_confidence, applied_overrides)
    """
    return safety_agent.apply_safety_overrides(text, detected_symptoms, base_prediction, base_confidence)

def generate_safety_message(prediction: str,
                           applied_overrides: List[Dict],
                           detected_symptoms: List[str]) -> str:
    """
    Generate safety message for the user
    
    Args:
        prediction: Final triage prediction
        applied_overrides: List of applied safety overrides
        detected_symptoms: List of detected symptoms
        
    Returns:
        Safety message string
    """
    return safety_agent.generate_safety_message(prediction, applied_overrides, detected_symptoms)