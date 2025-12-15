"""
Configuration settings for the Medical Symptoms Checker
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Model settings
TRIAGE_MODEL_PATH = MODELS_DIR / "triage_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
APRIORI_RULES_PATH = DATA_DIR / "association_rules.json"

# Triage levels
TRIAGE_LEVELS = {
    0: "self-care",
    1: "see_doctor", 
    2: "emergency"
}

TRIAGE_DESCRIPTIONS = {
    "self-care": "Monitor symptoms at home with basic care",
    "see_doctor": "Schedule an appointment with your doctor within 24-48 hours", 
    "emergency": "Seek immediate emergency medical attention"
}

# Symptom vocabulary - common symptoms and their variations
SYMPTOM_VOCABULARY = {
    "fever": ["fever", "high temperature", "hot", "burning up", "feverish"],
    "headache": ["headache", "head pain", "migraine", "head ache"],
    "cough": ["cough", "coughing", "hacking"],
    "chest_pain": ["chest pain", "chest ache", "heart pain", "chest pressure"],
    "shortness_breath": ["shortness of breath", "difficulty breathing", "breathless", "can't breathe"],
    "nausea": ["nausea", "nauseous", "sick to stomach", "queasy"],
    "vomiting": ["vomiting", "throwing up", "puking", "vomit"],
    "diarrhea": ["diarrhea", "loose stools", "watery stools"],
    "fatigue": ["fatigue", "tired", "exhausted", "weakness", "weak"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "vertigo"],
    "sore_throat": ["sore throat", "throat pain", "scratchy throat"],
    "runny_nose": ["runny nose", "stuffy nose", "congestion", "nasal congestion"],
    "muscle_pain": ["muscle pain", "body aches", "muscle aches", "joint pain"],
    "abdominal_pain": ["stomach pain", "belly pain", "abdominal pain", "tummy ache"]
}

# Red flag symptoms that require immediate attention
RED_FLAG_SYMPTOMS = [
    "severe chest pain",
    "difficulty breathing", 
    "sudden severe headache",
    "loss of consciousness",
    "severe bleeding",
    "signs of stroke",
    "severe allergic reaction"
]

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.8

# LLM settings (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_LLM_EXPLANATIONS = bool(OPENAI_API_KEY)

# Safety disclaimers
MEDICAL_DISCLAIMER = """
⚠️ IMPORTANT DISCLAIMER: This is not medical advice. This tool provides general information only. 
Always consult with a qualified healthcare professional for proper medical diagnosis and treatment. 
In case of emergency, call your local emergency services immediately.
"""