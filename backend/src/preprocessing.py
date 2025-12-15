"""
Preprocessing Agent - Handles text cleaning and symptom extraction
Responsibility: Convert raw user input into structured features for ML models
"""
import re
import string
import nltk
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .config import SYMPTOM_VOCABULARY

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class PreprocessingAgent:
    """
    Agent responsible for cleaning text and extracting symptoms from user input
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.symptom_vocab = SYMPTOM_VOCABULARY
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text
        
        Args:
            text: Raw user input text
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        # Handle common medical abbreviations
        text = text.replace("can't breathe", "difficulty breathing")
        text = text.replace("short of breath", "shortness of breath")
        text = text.replace("throwing up", "vomiting")
        
        return text.strip()
    
    def extract_symptoms(self, text: str) -> Dict[str, bool]:
        """
        Extract symptoms from text using keyword matching
        
        Args:
            text: Cleaned text input
            
        Returns:
            Dictionary with symptom flags
        """
        text_lower = text.lower()
        detected_symptoms = {}
        
        # Check each symptom category
        for symptom_key, variations in self.symptom_vocab.items():
            detected = False
            for variation in variations:
                if variation.lower() in text_lower:
                    detected = True
                    break
            detected_symptoms[symptom_key] = detected
            
        return detected_symptoms
    
    def extract_severity_indicators(self, text: str) -> Dict[str, int]:
        """
        Extract severity indicators from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with severity scores (0-3 scale)
        """
        text_lower = text.lower()
        severity_indicators = {
            'pain_severity': 0,
            'duration_severity': 0,
            'intensity_severity': 0
        }
        
        # Pain severity keywords
        if any(word in text_lower for word in ['severe', 'excruciating', 'unbearable']):
            severity_indicators['pain_severity'] = 3
        elif any(word in text_lower for word in ['moderate', 'bad', 'intense']):
            severity_indicators['pain_severity'] = 2
        elif any(word in text_lower for word in ['mild', 'slight', 'little']):
            severity_indicators['pain_severity'] = 1
            
        # Duration indicators
        if any(word in text_lower for word in ['weeks', 'months', 'chronic']):
            severity_indicators['duration_severity'] = 3
        elif any(word in text_lower for word in ['days', 'week']):
            severity_indicators['duration_severity'] = 2
        elif any(word in text_lower for word in ['hours', 'today', 'yesterday']):
            severity_indicators['duration_severity'] = 1
            
        # Intensity indicators
        if any(word in text_lower for word in ['getting worse', 'worsening', 'increasing']):
            severity_indicators['intensity_severity'] = 3
        elif any(word in text_lower for word in ['same', 'stable', 'constant']):
            severity_indicators['intensity_severity'] = 2
        elif any(word in text_lower for word in ['better', 'improving', 'decreasing']):
            severity_indicators['intensity_severity'] = 1
            
        return severity_indicators
    
    def create_feature_vector(self, text: str, age: Optional[int] = None, 
                            gender: Optional[str] = None) -> Tuple[Dict, List[str]]:
        """
        Create complete feature vector from user input
        
        Args:
            text: User symptom description
            age: User age (optional)
            gender: User gender (optional)
            
        Returns:
            Tuple of (feature_dict, detected_symptoms_list)
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract symptoms
        symptoms = self.extract_symptoms(cleaned_text)
        
        # Extract severity indicators
        severity = self.extract_severity_indicators(cleaned_text)
        
        # Create feature dictionary
        features = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            **symptoms,
            **severity,
            'age_group': self._categorize_age(age) if age else 'unknown',
            'gender': gender.lower() if gender else 'unknown',
            'symptom_count': sum(symptoms.values()),
            'text_length': len(cleaned_text.split())
        }
        
        # Get list of detected symptoms for rules engine
        detected_symptoms_list = [symptom for symptom, detected in symptoms.items() if detected]
        
        return features, detected_symptoms_list
    
    def _categorize_age(self, age: int) -> str:
        """Categorize age into groups"""
        if age < 18:
            return 'child'
        elif age < 65:
            return 'adult'
        else:
            return 'senior'
    
    def preprocess_for_training(self, df: pd.DataFrame, text_column: str = 'complaint_text') -> pd.DataFrame:
        """
        Preprocess dataset for model training
        
        Args:
            df: Training dataframe
            text_column: Name of text column
            
        Returns:
            Processed dataframe with feature columns
        """
        processed_data = []
        
        for _, row in df.iterrows():
            features, _ = self.create_feature_vector(
                row[text_column], 
                row.get('age'), 
                row.get('gender')
            )
            processed_data.append(features)
            
        return pd.DataFrame(processed_data)

# Global instance
preprocessing_agent = PreprocessingAgent()

def preprocess_input(text: str, age: Optional[int] = None, 
                    gender: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """
    Main preprocessing function to be called by API
    
    Args:
        text: User symptom description
        age: User age (optional)
        gender: User gender (optional)
        
    Returns:
        Tuple of (features_dict, detected_symptoms_list)
    """
    return preprocessing_agent.create_feature_vector(text, age, gender)