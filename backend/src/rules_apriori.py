"""
Rules and Apriori Agent - Handles association rules and pattern-based adjustments
Responsibility: Apply mined association rules and pattern-based logic to enhance predictions
"""
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from .config import APRIORI_RULES_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AprioriRulesAgent:
    """
    Agent responsible for applying association rules mined from symptom patterns
    """
    
    def __init__(self):
        self.association_rules = []
        self.symptom_patterns = {}
        self.is_loaded = False
        self._initialize_default_patterns()
        
    def _initialize_default_patterns(self):
        """
        Initialize default symptom patterns based on medical knowledge
        These serve as fallback rules when Apriori rules are not available
        """
        self.default_patterns = {
            # Emergency patterns
            "emergency_patterns": [
                {"symptoms": ["chest_pain", "shortness_breath"], "confidence": 0.9, "action": "emergency"},
                {"symptoms": ["severe_headache", "fever"], "confidence": 0.8, "action": "emergency"},
                {"symptoms": ["chest_pain", "dizziness"], "confidence": 0.85, "action": "emergency"},
                {"symptoms": ["shortness_breath", "chest_pain", "fatigue"], "confidence": 0.95, "action": "emergency"}
            ],
            
            # Doctor visit patterns  
            "doctor_patterns": [
                {"symptoms": ["fever", "cough", "fatigue"], "confidence": 0.7, "action": "see_doctor"},
                {"symptoms": ["headache", "fever", "muscle_pain"], "confidence": 0.75, "action": "see_doctor"},
                {"symptoms": ["abdominal_pain", "nausea", "vomiting"], "confidence": 0.8, "action": "see_doctor"},
                {"symptoms": ["sore_throat", "fever", "headache"], "confidence": 0.7, "action": "see_doctor"}
            ],
            
            # Self-care patterns
            "selfcare_patterns": [
                {"symptoms": ["runny_nose", "sore_throat"], "confidence": 0.6, "action": "self-care"},
                {"symptoms": ["mild_headache"], "confidence": 0.5, "action": "self-care"},
                {"symptoms": ["fatigue"], "confidence": 0.4, "action": "self-care"}
            ]
        }
    
    def load_apriori_rules(self) -> bool:
        """
        Load association rules from JSON file
        
        Returns:
            True if rules loaded successfully, False otherwise
        """
        try:
            if APRIORI_RULES_PATH.exists():
                with open(APRIORI_RULES_PATH, 'r') as f:
                    data = json.load(f)
                    self.association_rules = data.get('rules', [])
                    self.symptom_patterns = data.get('patterns', {})
                logger.info(f"Loaded {len(self.association_rules)} association rules")
                self.is_loaded = True
                return True
            else:
                logger.info("No Apriori rules file found, using default patterns")
                self.is_loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Error loading Apriori rules: {str(e)}")
            self.is_loaded = False
            return False
    
    def find_matching_patterns(self, detected_symptoms: List[str]) -> List[Dict]:
        """
        Find patterns that match the detected symptoms
        
        Args:
            detected_symptoms: List of detected symptom keys
            
        Returns:
            List of matching patterns with their confidence scores
        """
        matching_patterns = []
        detected_set = set(detected_symptoms)
        
        # Check loaded Apriori rules first
        if self.is_loaded and self.association_rules:
            for rule in self.association_rules:
                rule_symptoms = set(rule.get('antecedent', []))
                if rule_symptoms.issubset(detected_set):
                    matching_patterns.append({
                        'symptoms': list(rule_symptoms),
                        'consequent': rule.get('consequent'),
                        'confidence': rule.get('confidence', 0.5),
                        'support': rule.get('support', 0.1),
                        'lift': rule.get('lift', 1.0),
                        'source': 'apriori'
                    })
        
        # Check default patterns as fallback
        for pattern_type, patterns in self.default_patterns.items():
            for pattern in patterns:
                pattern_symptoms = set(pattern['symptoms'])
                if pattern_symptoms.issubset(detected_set):
                    matching_patterns.append({
                        'symptoms': pattern['symptoms'],
                        'action': pattern['action'],
                        'confidence': pattern['confidence'],
                        'source': 'default'
                    })
        
        # Sort by confidence (highest first)
        matching_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return matching_patterns
    
    def calculate_pattern_strength(self, detected_symptoms: List[str]) -> Dict[str, float]:
        """
        Calculate strength scores for different triage actions based on patterns
        
        Args:
            detected_symptoms: List of detected symptom keys
            
        Returns:
            Dictionary with action strengths
        """
        action_strengths = {
            'emergency': 0.0,
            'see_doctor': 0.0,
            'self-care': 0.0
        }
        
        matching_patterns = self.find_matching_patterns(detected_symptoms)
        
        for pattern in matching_patterns:
            action = pattern.get('action', pattern.get('consequent', 'see_doctor'))
            confidence = pattern.get('confidence', 0.5)
            
            # Map consequent to action if needed
            if action not in action_strengths:
                if 'emergency' in action.lower():
                    action = 'emergency'
                elif 'doctor' in action.lower():
                    action = 'see_doctor'
                else:
                    action = 'self-care'
            
            # Accumulate strength (weighted by confidence)
            if action in action_strengths:
                action_strengths[action] += confidence
        
        return action_strengths
    
    def apply_apriori_adjustment(self, detected_symptoms: List[str], 
                               base_prediction: str, 
                               base_confidence: float) -> Tuple[str, float, List[Dict]]:
        """
        Apply Apriori rules to adjust base prediction
        
        Args:
            detected_symptoms: List of detected symptom keys
            base_prediction: Original model prediction
            base_confidence: Original model confidence
            
        Returns:
            Tuple of (adjusted_prediction, adjusted_confidence, applied_rules)
        """
        if not detected_symptoms:
            return base_prediction, base_confidence, []
        
        # Calculate pattern strengths
        pattern_strengths = self.calculate_pattern_strength(detected_symptoms)
        matching_patterns = self.find_matching_patterns(detected_symptoms)
        
        # Find strongest pattern
        strongest_action = max(pattern_strengths.items(), key=lambda x: x[1])
        strongest_strength = strongest_action[1]
        
        applied_rules = []
        adjusted_prediction = base_prediction
        adjusted_confidence = base_confidence
        
        # Apply adjustment if pattern strength is significant
        if strongest_strength > 0.5:  # Threshold for pattern significance
            suggested_action = strongest_action[0]
            
            # Only upgrade severity, never downgrade for safety
            severity_order = {'self-care': 0, 'see_doctor': 1, 'emergency': 2}
            
            base_severity = severity_order.get(base_prediction, 1)
            pattern_severity = severity_order.get(suggested_action, 1)
            
            if pattern_severity > base_severity:
                adjusted_prediction = suggested_action
                # Boost confidence if pattern strongly suggests higher severity
                adjusted_confidence = min(0.95, base_confidence + (strongest_strength * 0.2))
                
                # Record applied rules
                for pattern in matching_patterns[:3]:  # Top 3 patterns
                    if pattern.get('action') == suggested_action or pattern.get('consequent') == suggested_action:
                        applied_rules.append({
                            'symptoms': pattern['symptoms'],
                            'action': pattern.get('action', pattern.get('consequent')),
                            'confidence': pattern['confidence'],
                            'source': pattern['source']
                        })
        
        return adjusted_prediction, adjusted_confidence, applied_rules
    
    def get_symptom_associations(self, symptom: str) -> List[Dict]:
        """
        Get symptoms commonly associated with the given symptom
        
        Args:
            symptom: Symptom key to find associations for
            
        Returns:
            List of associated symptoms with confidence scores
        """
        associations = []
        
        # Check Apriori rules
        if self.is_loaded and self.association_rules:
            for rule in self.association_rules:
                antecedent = rule.get('antecedent', [])
                consequent = rule.get('consequent', [])
                
                if symptom in antecedent:
                    for assoc_symptom in antecedent + [consequent] if isinstance(consequent, str) else antecedent + consequent:
                        if assoc_symptom != symptom:
                            associations.append({
                                'symptom': assoc_symptom,
                                'confidence': rule.get('confidence', 0.5),
                                'support': rule.get('support', 0.1)
                            })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_associations = []
        for assoc in associations:
            if assoc['symptom'] not in seen:
                seen.add(assoc['symptom'])
                unique_associations.append(assoc)
        
        unique_associations.sort(key=lambda x: x['confidence'], reverse=True)
        return unique_associations[:5]  # Top 5 associations

# Global instance
apriori_agent = AprioriRulesAgent()

def load_apriori_rules() -> bool:
    """
    Load Apriori rules at startup
    
    Returns:
        True if successful, False otherwise
    """
    return apriori_agent.load_apriori_rules()

def apply_apriori_rules(detected_symptoms: List[str], 
                       base_prediction: str, 
                       base_confidence: float) -> Tuple[str, float, List[Dict]]:
    """
    Main function to apply Apriori rules adjustment
    
    Args:
        detected_symptoms: List of detected symptom keys
        base_prediction: Original model prediction
        base_confidence: Original model confidence
        
    Returns:
        Tuple of (adjusted_prediction, adjusted_confidence, applied_rules)
    """
    return apriori_agent.apply_apriori_adjustment(detected_symptoms, base_prediction, base_confidence)

def get_symptom_associations(symptom: str) -> List[Dict]:
    """
    Get symptoms associated with the given symptom
    
    Args:
        symptom: Symptom to find associations for
        
    Returns:
        List of associated symptoms
    """
    return apriori_agent.get_symptom_associations(symptom)