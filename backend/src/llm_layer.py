"""
LLM Layer Agent - Handles LLM-based explanations and user-friendly text generation
Responsibility: Generate explanations and user-friendly text without making medical decisions
"""
import os
import logging
from typing import Dict, List, Optional
from .config import OPENAI_API_KEY, USE_LLM_EXPLANATIONS, MEDICAL_DISCLAIMER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    if USE_LLM_EXPLANATIONS:
        import openai
        openai.api_key = OPENAI_API_KEY
    else:
        openai = None
except ImportError:
    openai = None
    logger.warning("OpenAI library not available. LLM explanations disabled.")

class LLMLayerAgent:
    """
    Agent responsible for generating user-friendly explanations using LLM
    IMPORTANT: This agent NEVER makes medical decisions, only explains existing decisions
    """
    
    def __init__(self):
        self.client = None
        self.is_available = False
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OpenAI client if available"""
        if openai and OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                self.is_available = True
                logger.info("LLM client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {str(e)}")
                self.is_available = False
        else:
            logger.info("LLM not available - using template-based explanations")
            self.is_available = False
    
    def generate_explanation(self, 
                           detected_symptoms: List[str],
                           prediction: str,
                           confidence: float,
                           applied_rules: Optional[List[Dict]] = None) -> str:
        """
        Generate user-friendly explanation of the triage decision
        
        Args:
            detected_symptoms: List of detected symptom keys
            prediction: Triage prediction (self-care/see_doctor/emergency)
            confidence: Prediction confidence
            applied_rules: List of applied rules/overrides
            
        Returns:
            User-friendly explanation string
        """
        if self.is_available:
            try:
                return self._generate_llm_explanation(detected_symptoms, prediction, confidence, applied_rules)
            except Exception as e:
                logger.error(f"LLM explanation failed: {str(e)}")
                return self._generate_template_explanation(detected_symptoms, prediction, confidence)
        else:
            return self._generate_template_explanation(detected_symptoms, prediction, confidence)
    
    def _generate_llm_explanation(self,
                                detected_symptoms: List[str],
                                prediction: str,
                                confidence: float,
                                applied_rules: Optional[List[Dict]] = None) -> str:
        """
        Generate explanation using LLM
        
        Args:
            detected_symptoms: List of detected symptom keys
            prediction: Triage prediction
            confidence: Prediction confidence
            applied_rules: Applied rules/overrides
            
        Returns:
            LLM-generated explanation
        """
        # Convert symptom keys to readable names
        symptom_names = self._convert_symptoms_to_readable(detected_symptoms)
        
        # Create prompt with strict constraints
        prompt = f"""
You are helping explain a medical triage recommendation to a user. You must follow these strict rules:

1. NEVER diagnose any medical condition
2. NEVER recommend specific treatments
3. NEVER contradict the given triage recommendation
4. Always include that this is not medical advice
5. Keep explanation general and educational only

Given information:
- Detected symptoms: {', '.join(symptom_names) if symptom_names else 'None specified'}
- Recommendation: {prediction.replace('_', ' ')}
- Confidence level: {confidence:.1%}

Provide a brief, supportive explanation (2-3 sentences) of why this recommendation was made based on the symptoms, without diagnosing or giving medical advice. Focus on general health guidance and the importance of professional medical care when needed.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains medical triage recommendations without providing medical advice or diagnoses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Add safety disclaimer
            explanation += f"\n\n{MEDICAL_DISCLAIMER}"
            
            return explanation
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise e
    
    def _generate_template_explanation(self,
                                     detected_symptoms: List[str],
                                     prediction: str,
                                     confidence: float) -> str:
        """
        Generate explanation using templates (fallback when LLM not available)
        
        Args:
            detected_symptoms: List of detected symptom keys
            prediction: Triage prediction
            confidence: Prediction confidence
            
        Returns:
            Template-based explanation
        """
        symptom_names = self._convert_symptoms_to_readable(detected_symptoms)
        symptom_count = len(detected_symptoms)
        
        # Base explanations for each prediction type
        base_explanations = {
            "self-care": [
                "Based on the symptoms you've described, home care and monitoring may be appropriate.",
                "Your symptoms appear to be mild and may improve with rest and basic care.",
                "The symptoms you've mentioned are commonly manageable at home with proper self-care."
            ],
            "see_doctor": [
                "Your symptoms suggest it would be wise to consult with a healthcare professional.",
                "The combination of symptoms you've described warrants medical evaluation.",
                "Based on what you've shared, a medical professional should assess your condition."
            ],
            "emergency": [
                "Your symptoms indicate you should seek immediate medical attention.",
                "The symptoms you've described require urgent medical evaluation.",
                "Based on your symptoms, emergency medical care is recommended."
            ]
        }
        
        # Select base explanation
        explanations = base_explanations.get(prediction, base_explanations["see_doctor"])
        base_explanation = explanations[0]  # Use first explanation as default
        
        # Add symptom-specific context
        context_parts = []
        
        if symptom_count > 1:
            context_parts.append(f"You've reported {symptom_count} different symptoms")
        
        if symptom_names:
            if len(symptom_names) <= 3:
                context_parts.append(f"including {', '.join(symptom_names)}")
            else:
                context_parts.append(f"including {', '.join(symptom_names[:3])} and others")
        
        # Add confidence context
        if confidence >= 0.8:
            confidence_text = "This recommendation is made with high confidence"
        elif confidence >= 0.6:
            confidence_text = "This recommendation is made with moderate confidence"
        else:
            confidence_text = "This recommendation is made with caution due to uncertainty"
        
        # Combine explanation parts
        full_explanation = base_explanation
        
        if context_parts:
            full_explanation += f" {', '.join(context_parts)}."
        
        full_explanation += f" {confidence_text} based on the information provided."
        
        # Add specific guidance based on prediction
        if prediction == "emergency":
            full_explanation += " Please do not delay in seeking medical care."
        elif prediction == "see_doctor":
            full_explanation += " Consider scheduling an appointment soon to discuss your symptoms."
        elif prediction == "self-care":
            full_explanation += " However, if symptoms worsen or persist, don't hesitate to seek medical advice."
        
        # Add medical disclaimer
        full_explanation += f"\n\n{MEDICAL_DISCLAIMER}"
        
        return full_explanation
    
    def _convert_symptoms_to_readable(self, symptom_keys: List[str]) -> List[str]:
        """
        Convert symptom keys to readable names
        
        Args:
            symptom_keys: List of symptom keys (e.g., ['chest_pain', 'fever'])
            
        Returns:
            List of readable symptom names
        """
        readable_mapping = {
            'fever': 'fever',
            'headache': 'headache',
            'cough': 'cough',
            'chest_pain': 'chest pain',
            'shortness_breath': 'shortness of breath',
            'nausea': 'nausea',
            'vomiting': 'vomiting',
            'diarrhea': 'diarrhea',
            'fatigue': 'fatigue',
            'dizziness': 'dizziness',
            'sore_throat': 'sore throat',
            'runny_nose': 'runny nose',
            'muscle_pain': 'muscle pain',
            'abdominal_pain': 'abdominal pain'
        }
        
        return [readable_mapping.get(key, key.replace('_', ' ')) for key in symptom_keys]
    
    def generate_follow_up_questions(self, detected_symptoms: List[str]) -> List[str]:
        """
        Generate follow-up questions to clarify symptoms
        
        Args:
            detected_symptoms: List of detected symptom keys
            
        Returns:
            List of follow-up questions
        """
        questions = []
        
        # General questions based on detected symptoms
        if 'fever' in detected_symptoms:
            questions.append("How high is your temperature, and how long have you had the fever?")
        
        if 'chest_pain' in detected_symptoms:
            questions.append("Can you describe the chest pain? Is it sharp, dull, or crushing?")
        
        if 'headache' in detected_symptoms:
            questions.append("How severe is the headache on a scale of 1-10?")
        
        if 'abdominal_pain' in detected_symptoms:
            questions.append("Where exactly is the abdominal pain located?")
        
        # General questions if no specific symptoms
        if not detected_symptoms:
            questions.extend([
                "Can you describe your main symptoms in more detail?",
                "How long have you been experiencing these symptoms?",
                "Have the symptoms been getting better, worse, or staying the same?"
            ])
        
        # Always include duration question if not already asked
        if not any('long' in q for q in questions):
            questions.append("How long have you been experiencing these symptoms?")
        
        return questions[:3]  # Return max 3 questions

# Global instance
llm_agent = LLMLayerAgent()

def generate_explanation(detected_symptoms: List[str],
                        prediction: str,
                        confidence: float,
                        applied_rules: Optional[List[Dict]] = None) -> str:
    """
    Main function to generate explanation
    
    Args:
        detected_symptoms: List of detected symptom keys
        prediction: Triage prediction
        confidence: Prediction confidence
        applied_rules: Applied rules/overrides
        
    Returns:
        User-friendly explanation
    """
    return llm_agent.generate_explanation(detected_symptoms, prediction, confidence, applied_rules)

def generate_follow_up_questions(detected_symptoms: List[str]) -> List[str]:
    """
    Generate follow-up questions for symptom clarification
    
    Args:
        detected_symptoms: List of detected symptom keys
        
    Returns:
        List of follow-up questions
    """
    return llm_agent.generate_follow_up_questions(detected_symptoms)

def is_llm_available() -> bool:
    """
    Check if LLM is available for use
    
    Returns:
        True if LLM is available, False otherwise
    """
    return llm_agent.is_available