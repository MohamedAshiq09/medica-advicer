"""
Medical Knowledge Base - Real medical data and decision rules
Provides evidence-based triage recommendations based on medical guidelines
"""
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriageLevel(Enum):
    SELF_CARE = "self-care"
    SEE_DOCTOR = "see_doctor"
    EMERGENCY = "emergency"

# Medical knowledge base with real clinical data
MEDICAL_CONDITIONS = {
    # Common Cold / URI
    "common_cold": {
        "symptoms": ["runny_nose", "cough", "sore_throat", "sneezing"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.85,
        "advice": "Rest, fluids, over-the-counter pain relievers. Symptoms typically resolve in 7-10 days.",
        "red_flags": ["high_fever", "severe_headache", "difficulty_breathing"],
        "duration": "7-10 days"
    },
    
    # Mild Fever
    "mild_fever": {
        "symptoms": ["fever"],
        "severity_threshold": 38.5,  # Celsius
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.80,
        "advice": "Rest, hydration, fever-reducing medication (acetaminophen/ibuprofen). Monitor temperature.",
        "red_flags": ["fever_with_severe_headache", "fever_with_stiff_neck", "fever_with_rash"],
        "duration": "3-5 days",
        "severity_score": 1.5
    },
    
    # Fever + Headache (Mild)
    "fever_headache_mild": {
        "symptoms": ["fever", "headache"],
        "severity_threshold": 38.5,
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.75,
        "advice": "Rest, hydration, pain relievers. Most viral infections resolve on their own.",
        "red_flags": ["severe_headache", "stiff_neck", "confusion", "rash"],
        "duration": "3-7 days",
        "severity_score": 1.5
    },
    
    # Fever + Headache (Severe) - Possible Meningitis
    "fever_headache_severe": {
        "symptoms": ["fever", "severe_headache", "neck_stiffness"],
        "triage": TriageLevel.EMERGENCY,
        "confidence": 0.95,
        "advice": "SEEK IMMEDIATE MEDICAL ATTENTION. Possible meningitis or other serious infection.",
        "red_flags": ["fever", "severe_headache", "neck_stiffness", "confusion", "rash"],
        "duration": "Immediate"
    },
    
    # Chest Pain
    "chest_pain": {
        "symptoms": ["chest_pain"],
        "triage": TriageLevel.EMERGENCY,
        "confidence": 0.90,
        "advice": "SEEK IMMEDIATE MEDICAL ATTENTION. Call emergency services.",
        "red_flags": ["chest_pain", "shortness_breath", "dizziness", "sweating"],
        "duration": "Immediate"
    },
    
    # Chest Pain + Shortness of Breath
    "chest_pain_sob": {
        "symptoms": ["chest_pain", "shortness_breath"],
        "triage": TriageLevel.EMERGENCY,
        "confidence": 0.98,
        "advice": "CALL EMERGENCY SERVICES IMMEDIATELY. Possible heart attack or pulmonary embolism.",
        "red_flags": ["chest_pain", "shortness_breath", "dizziness", "sweating", "nausea"],
        "duration": "Immediate"
    },
    
    # Shortness of Breath
    "shortness_breath": {
        "symptoms": ["shortness_breath"],
        "triage": TriageLevel.EMERGENCY,
        "confidence": 0.85,
        "advice": "SEEK IMMEDIATE MEDICAL ATTENTION. Breathing difficulties require urgent evaluation.",
        "red_flags": ["shortness_breath", "chest_pain", "dizziness"],
        "duration": "Immediate"
    },
    
    # Gastroenteritis (Mild)
    "gastroenteritis_mild": {
        "symptoms": ["nausea", "vomiting", "diarrhea"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.80,
        "advice": "Rest, clear fluids, electrolyte replacement. Avoid solid food initially.",
        "red_flags": ["severe_abdominal_pain", "blood_in_stool", "severe_dehydration"],
        "duration": "1-3 days"
    },
    
    # Severe Abdominal Pain
    "severe_abdominal_pain": {
        "symptoms": ["severe_abdominal_pain"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.85,
        "advice": "See a doctor promptly. Severe abdominal pain requires evaluation.",
        "red_flags": ["severe_abdominal_pain", "vomiting", "fever", "blood_in_stool"],
        "duration": "Same day",
        "severity_score": 2.0
    },
    
    # Cough (Mild)
    "cough_mild": {
        "symptoms": ["cough"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.75,
        "advice": "Rest, fluids, honey, cough drops. Most coughs resolve on their own.",
        "red_flags": ["persistent_cough", "cough_with_blood", "shortness_breath"],
        "duration": "1-3 weeks"
    },
    
    # Persistent Cough
    "cough_persistent": {
        "symptoms": ["persistent_cough"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.80,
        "advice": "See a doctor. Persistent cough may indicate infection or other condition.",
        "red_flags": ["cough_with_blood", "shortness_breath", "chest_pain"],
        "duration": "Same week",
        "severity_score": 1.5
    },
    
    # Sore Throat (Mild)
    "sore_throat_mild": {
        "symptoms": ["sore_throat"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.80,
        "advice": "Rest, warm liquids, throat lozenges, pain relievers.",
        "red_flags": ["high_fever", "difficulty_swallowing", "swollen_lymph_nodes"],
        "duration": "3-7 days"
    },
    
    # Strep Throat
    "strep_throat": {
        "symptoms": ["sore_throat", "fever", "swollen_lymph_nodes"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.85,
        "advice": "See a doctor for throat culture/rapid test. May need antibiotics.",
        "red_flags": ["high_fever", "difficulty_swallowing", "rash"],
        "duration": "Same day"
    },
    
    # Migraine
    "migraine": {
        "symptoms": ["severe_headache", "nausea"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.75,
        "advice": "Rest in dark, quiet room. Pain relievers, hydration. See doctor if severe.",
        "red_flags": ["worst_headache_of_life", "fever", "stiff_neck", "confusion"],
        "duration": "4-72 hours"
    },
    
    # Dizziness (Mild)
    "dizziness_mild": {
        "symptoms": ["dizziness"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.70,
        "advice": "Rest, avoid sudden movements, stay hydrated.",
        "red_flags": ["dizziness_with_chest_pain", "dizziness_with_shortness_breath"],
        "duration": "1-3 days"
    },
    
    # Dizziness (Severe)
    "dizziness_severe": {
        "symptoms": ["severe_dizziness", "loss_of_balance"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.80,
        "advice": "See a doctor. Severe dizziness may indicate inner ear or neurological issue.",
        "red_flags": ["dizziness_with_chest_pain", "dizziness_with_shortness_breath"],
        "duration": "Same day"
    },
    
    # Rash (Mild)
    "rash_mild": {
        "symptoms": ["rash"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.75,
        "advice": "Keep area clean and dry. Avoid irritants. Use moisturizer if needed.",
        "red_flags": ["rash_with_fever", "spreading_rash", "rash_with_difficulty_breathing"],
        "duration": "1-2 weeks"
    },
    
    # Rash with Fever (Possible Measles/Chickenpox)
    "rash_with_fever": {
        "symptoms": ["rash", "fever"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.85,
        "advice": "See a doctor. Rash with fever may indicate contagious disease.",
        "red_flags": ["spreading_rash", "high_fever", "difficulty_breathing"],
        "duration": "Same day"
    },
    
    # Fatigue (Mild)
    "fatigue_mild": {
        "symptoms": ["fatigue"],
        "triage": TriageLevel.SELF_CARE,
        "confidence": 0.80,
        "advice": "Rest, sleep, nutrition. Fatigue often improves with rest.",
        "red_flags": ["fatigue_with_fever", "fatigue_with_chest_pain", "persistent_fatigue"],
        "duration": "1-2 weeks"
    },
    
    # Persistent Fatigue
    "fatigue_persistent": {
        "symptoms": ["persistent_fatigue"],
        "triage": TriageLevel.SEE_DOCTOR,
        "confidence": 0.80,
        "advice": "See a doctor. Persistent fatigue may indicate underlying condition.",
        "red_flags": ["fatigue_with_fever", "fatigue_with_weight_loss"],
        "duration": "Same week"
    },
}

# Symptom severity levels
SYMPTOM_SEVERITY = {
    # High severity (always escalate)
    "chest_pain": 3,
    "shortness_breath": 3,
    "severe_headache": 3,
    "loss_of_consciousness": 3,
    "severe_bleeding": 3,
    "difficulty_breathing": 3,
    "worst_headache_of_life": 3,
    "stiff_neck": 3,
    "confusion": 3,
    "seizure": 3,
    
    # Medium severity
    "fever": 2,
    "headache": 2,
    "vomiting": 2,
    "diarrhea": 2,
    "abdominal_pain": 2,
    "dizziness": 2,
    "high_fever": 2,
    "severe_dizziness": 2,
    "severe_abdominal_pain": 2,
    "persistent_cough": 2,
    
    # Low severity
    "runny_nose": 1,
    "cough": 1,
    "sore_throat": 1,
    "sneezing": 1,
    "fatigue": 1,
    "rash": 1,
    "muscle_pain": 1,
    "nausea": 1,
    "swollen_lymph_nodes": 1,
}

# Symptom combinations that indicate emergency
EMERGENCY_COMBINATIONS = [
    ["chest_pain", "shortness_breath"],
    ["chest_pain", "dizziness"],
    ["fever", "severe_headache", "stiff_neck"],
    ["fever", "severe_headache", "confusion"],
    ["severe_headache", "vision_loss"],
    ["severe_headache", "weakness"],
    ["shortness_breath", "chest_pain"],
    ["shortness_breath", "dizziness"],
    ["loss_of_consciousness"],
    ["severe_bleeding"],
    ["seizure"],
]

class MedicalKnowledgeBase:
    """
    Medical knowledge base for evidence-based triage
    """
    
    def __init__(self):
        self.conditions = MEDICAL_CONDITIONS
        self.severity_map = SYMPTOM_SEVERITY
        self.emergency_combos = EMERGENCY_COMBINATIONS
    
    def find_matching_condition(self, detected_symptoms: List[str]) -> Tuple[Optional[str], Dict]:
        """
        Find matching medical condition from knowledge base
        
        Args:
            detected_symptoms: List of detected symptoms
            
        Returns:
            Tuple of (condition_name, condition_data)
        """
        detected_set = set(detected_symptoms)
        best_match = None
        best_score = 0
        
        for condition_name, condition_data in self.conditions.items():
            condition_symptoms = set(condition_data.get("symptoms", []))
            
            if not condition_symptoms:
                continue
            
            # Calculate match score (Jaccard similarity)
            intersection = len(detected_set & condition_symptoms)
            union = len(detected_set | condition_symptoms)
            
            if union > 0:
                score = intersection / union
                
                # Boost score for exact matches
                if detected_set == condition_symptoms:
                    score = 1.0
                
                if score > best_score:
                    best_score = score
                    best_match = (condition_name, condition_data)
        
        if best_score > 0.5:  # At least 50% match
            return best_match
        
        return None, {}
    
    def check_emergency_symptoms(self, detected_symptoms: List[str]) -> Tuple[bool, str]:
        """
        Check if symptoms indicate emergency
        
        Args:
            detected_symptoms: List of detected symptoms
            
        Returns:
            Tuple of (is_emergency, reason)
        """
        detected_set = set(detected_symptoms)
        
        # Check emergency combinations
        for combo in self.emergency_combos:
            if set(combo).issubset(detected_set):
                return True, f"Emergency symptoms detected: {', '.join(combo)}"
        
        # Check high-severity individual symptoms
        high_severity = [s for s in detected_symptoms if self.severity_map.get(s, 0) == 3]
        if high_severity:
            return True, f"High-severity symptoms: {', '.join(high_severity)}"
        
        return False, ""
    
    def get_triage_recommendation(self, detected_symptoms: List[str]) -> Dict:
        """
        Get triage recommendation based on symptoms
        
        Args:
            detected_symptoms: List of detected symptoms
            
        Returns:
            Dictionary with triage recommendation
        """
        # Check for emergency first
        is_emergency, emergency_reason = self.check_emergency_symptoms(detected_symptoms)
        if is_emergency:
            return {
                "triage_level": TriageLevel.EMERGENCY.value,
                "confidence": 0.95,
                "reason": emergency_reason,
                "advice": "SEEK IMMEDIATE MEDICAL ATTENTION. Call emergency services.",
                "matched_condition": None
            }
        
        # Find matching condition
        condition_name, condition_data = self.find_matching_condition(detected_symptoms)
        
        if condition_name:
            return {
                "triage_level": condition_data["triage"].value,
                "confidence": condition_data.get("confidence", 0.70),
                "reason": f"Matched condition: {condition_name.replace('_', ' ').title()}",
                "advice": condition_data.get("advice", "Consult with healthcare provider"),
                "matched_condition": condition_name,
                "red_flags": condition_data.get("red_flags", []),
                "duration": condition_data.get("duration", "Unknown")
            }
        
        # Default: conservative recommendation
        return {
            "triage_level": TriageLevel.SEE_DOCTOR.value,
            "confidence": 0.60,
            "reason": "Symptoms require medical evaluation",
            "advice": "See a healthcare provider for proper diagnosis",
            "matched_condition": None
        }
    
    def get_symptom_severity(self, symptom: str) -> int:
        """
        Get severity level of a symptom (0-3)
        
        Args:
            symptom: Symptom name
            
        Returns:
            Severity level (0=unknown, 1=low, 2=medium, 3=high)
        """
        return self.severity_map.get(symptom, 0)
    
    def get_average_severity(self, symptoms: List[str]) -> float:
        """
        Get average severity of symptoms
        
        Args:
            symptoms: List of symptoms
            
        Returns:
            Average severity (0-3)
        """
        if not symptoms:
            return 0
        
        total = sum(self.get_symptom_severity(s) for s in symptoms)
        return total / len(symptoms)
    
    def get_condition_info(self, condition_name: str) -> Dict:
        """
        Get detailed information about a condition
        
        Args:
            condition_name: Name of condition
            
        Returns:
            Condition information
        """
        return self.conditions.get(condition_name, {})

# Global instance
knowledge_base = MedicalKnowledgeBase()

def get_triage_recommendation(symptoms: List[str]) -> Dict:
    """Get triage recommendation from knowledge base"""
    return knowledge_base.get_triage_recommendation(symptoms)

def check_emergency(symptoms: List[str]) -> Tuple[bool, str]:
    """Check if symptoms indicate emergency"""
    return knowledge_base.check_emergency_symptoms(symptoms)

def get_symptom_severity(symptom: str) -> int:
    """Get severity of a symptom"""
    return knowledge_base.get_symptom_severity(symptom)
