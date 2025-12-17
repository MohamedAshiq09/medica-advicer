#!/usr/bin/env python3
"""
Test the age/gender-aware triage system end-to-end
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.improved_model_inference import predict_triage as ml_predict
from src.enhanced_triage import perform_triage
from src.preprocessing import preprocess_input

def test_case(description, text, age, gender, expected_level):
    """Test a single case"""
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"Input: Age {age}, Gender {gender}")
    print(f"Symptoms: {text}")
    print(f"Expected: {expected_level}")
    print(f"{'-'*70}")
    
    # ML prediction
    ml_label, ml_probs, ml_conf = ml_predict(text, age, gender)
    print(f"ML Prediction: {ml_label} (confidence: {ml_conf:.2f})")
    print(f"  Probabilities: {ml_probs}")
    
    # Preprocessing
    features, symptoms = preprocess_input(text, age, gender)
    print(f"Detected symptoms: {symptoms}")
    
    # Enhanced triage
    result = perform_triage(symptoms, text, age, gender)
    print(f"\nFinal Triage: {result['triage_level']} (confidence: {result['confidence']:.2f})")
    print(f"Reasoning: {result['reason']}")
    print(f"Severity: {result['severity']}")
    
    # Check if correct
    is_correct = result['triage_level'] == expected_level
    status = "✅ PASS" if is_correct else "❌ FAIL"
    print(f"\n{status}")
    
    return is_correct

def main():
    print("🏥 Age and Gender-Aware Triage System - Comprehensive Test")
    print("="*70)
    
    test_cases = [
        # Young adults with mild symptoms
        ("Young adult with mild headache", 
         "I have a mild headache and feel a bit tired", 25, 'F', 'self-care'),
        
        # Young adult with severe symptoms
        ("Young adult with severe chest pain",
         "Severe chest pain and can't breathe properly", 45, 'M', 'emergency'),
        
        # Elderly with chest pain (high risk)
        ("Elderly male with chest pain",
         "Chest pain in elderly patient with shortness of breath", 72, 'M', 'emergency'),
        
        # Young child with fever
        ("Young child with fever",
         "Fever in young child with cough", 4, 'M', 'see_doctor'),
        
        # Elderly with fever
        ("Elderly with fever",
         "Fever and confusion in elderly patient", 78, 'M', 'see_doctor'),
        
        # Mild cold symptoms
        ("Young adult with cold",
         "Mild cold symptoms, runny nose", 24, 'F', 'self-care'),
        
        # Severe headache (worst of life)
        ("Severe headache - worst of life",
         "Worst headache of my life, sudden onset", 38, 'M', 'emergency'),
        
        # Abdominal pain with blood
        ("Severe abdominal pain with blood",
         "Severe abdominal pain with blood in stool", 52, 'M', 'emergency'),
        
        # Elderly with joint pain
        ("Elderly with joint pain",
         "Severe joint pain in elderly", 87, 'M', 'see_doctor'),
        
        # Young adult with nausea
        ("Young adult with mild nausea",
         "Nausea in young adult", 25, 'M', 'self-care'),
        
        # Elderly with severe fatigue
        ("Elderly with severe fatigue",
         "Severe fatigue in elderly", 88, 'M', 'see_doctor'),
        
        # Infant with fever (very high risk)
        ("Infant with fever",
         "Fever in infant with lethargy", 1, 'F', 'emergency'),
        
        # Sore throat in young adult
        ("Young adult with sore throat",
         "Mild sore throat in young adult", 21, 'F', 'self-care'),
        
        # Severe sore throat with fever
        ("Severe sore throat with fever",
         "Severe sore throat with fever in 35 year old", 35, 'F', 'see_doctor'),
        
        # Difficulty breathing
        ("Difficulty breathing",
         "Difficulty breathing in young adult", 26, 'M', 'emergency'),
    ]
    
    results = []
    for test_case_data in test_cases:
        result = test_case(*test_case_data)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage >= 80:
        print("✅ System is working well!")
    elif percentage >= 60:
        print("⚠️  System needs improvement")
    else:
        print("❌ System needs significant work")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
