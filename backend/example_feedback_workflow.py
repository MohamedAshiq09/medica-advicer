"""
Example workflow showing how to use the feedback learning system
This demonstrates the complete cycle: predict -> get feedback -> retrain
"""
import sys
sys.path.insert(0, '/backend')

from src.preprocessing import preprocess_input
from src.model_inference import predict_triage
from src.feedback_learning import (
    record_user_feedback,
    get_feedback_statistics,
    should_retrain_model,
    retrain_from_feedback
)

def example_workflow():
    """
    Example: User has fever and headache
    Model predicts 100% "visit doctor"
    User says they can self-manage
    System learns and improves
    """
    
    print("=" * 60)
    print("FEEDBACK LEARNING WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Step 1: User input
    user_input = "I have a fever and headache, but it's mild. I've had this before and managed it at home."
    print(f"\n1. USER INPUT: {user_input}")
    
    # Step 2: Preprocess
    features_dict, detected_symptoms = preprocess_input(user_input, age=30, gender="M")
    print(f"\n2. DETECTED SYMPTOMS: {detected_symptoms}")
    
    # Step 3: Get model prediction
    predicted_label, probabilities, confidence = predict_triage(features_dict)
    print(f"\n3. MODEL PREDICTION:")
    print(f"   - Label: {predicted_label}")
    print(f"   - Confidence: {confidence:.2%}")
    print(f"   - Probabilities: {probabilities}")
    
    # Step 4: User provides feedback
    user_correction = "self-care"  # User says they can self-manage
    user_notes = "Mild fever, I've handled this before with rest and fluids"
    
    print(f"\n4. USER FEEDBACK:")
    print(f"   - Correct label: {user_correction}")
    print(f"   - Notes: {user_notes}")
    
    # Step 5: Record feedback
    success = record_user_feedback(
        symptoms=detected_symptoms,
        predicted_label=predicted_label,
        user_correction=user_correction,
        user_notes=user_notes,
        severity_info={"severity": "mild", "duration": "2 days"}
    )
    print(f"\n5. FEEDBACK RECORDED: {success}")
    
    # Step 6: Check feedback statistics
    stats = get_feedback_statistics()
    print(f"\n6. FEEDBACK STATISTICS:")
    print(f"   - Total feedback: {stats['total_feedback']}")
    print(f"   - Accuracy: {stats['accuracy_from_feedback']:.2%}")
    print(f"   - By label: {stats['by_label']}")
    
    # Step 7: Check if retraining is needed
    needs_retrain = should_retrain_model()
    print(f"\n7. RETRAIN NEEDED: {needs_retrain}")
    
    if needs_retrain:
        print("\n8. RETRAINING MODEL...")
        result = retrain_from_feedback()
        print(f"   - Success: {result.get('success')}")
        print(f"   - Model type: {result.get('model_type')}")
        print(f"   - CV Score: {result.get('cv_score', 'N/A')}")
        print(f"   - Training samples: {result.get('training_samples')}")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

def simulate_multiple_cases():
    """
    Simulate multiple cases to build up feedback data
    """
    print("\n" + "=" * 60)
    print("SIMULATING MULTIPLE FEEDBACK CASES")
    print("=" * 60)
    
    cases = [
        {
            "input": "I have a mild fever and headache",
            "symptoms": ["fever", "headache"],
            "predicted": "see_doctor",
            "correct": "self-care",
            "notes": "Mild symptoms, managed at home"
        },
        {
            "input": "Severe chest pain and shortness of breath",
            "symptoms": ["chest_pain", "shortness_breath"],
            "predicted": "see_doctor",
            "correct": "emergency",
            "notes": "Severe symptoms, went to ER"
        },
        {
            "input": "Slight cough and runny nose",
            "symptoms": ["cough", "runny_nose"],
            "predicted": "self-care",
            "correct": "self-care",
            "notes": "Common cold, resolved with rest"
        },
        {
            "input": "High fever, severe headache, stiff neck",
            "symptoms": ["fever", "headache", "neck_stiffness"],
            "predicted": "see_doctor",
            "correct": "emergency",
            "notes": "Possible meningitis, hospitalized"
        }
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\nCase {i}: {case['input']}")
        
        success = record_user_feedback(
            symptoms=case['symptoms'],
            predicted_label=case['predicted'],
            user_correction=case['correct'],
            user_notes=case['notes']
        )
        
        was_correct = case['predicted'] == case['correct']
        print(f"  Prediction was {'CORRECT' if was_correct else 'INCORRECT'}")
        print(f"  Feedback recorded: {success}")
    
    # Check final stats
    stats = get_feedback_statistics()
    print(f"\n\nFINAL STATISTICS:")
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Overall accuracy: {stats['accuracy_from_feedback']:.2%}")
    print(f"  By label: {stats['by_label']}")
    
    # Check if retraining is needed
    if should_retrain_model():
        print(f"\n  Retraining needed: YES")
        result = retrain_from_feedback()
        if result.get('success'):
            print(f"  Retrain successful!")
            print(f"  New model type: {result.get('model_type')}")
            print(f"  CV Score: {result.get('cv_score'):.3f}")
    else:
        print(f"\n  Retraining needed: NO")

if __name__ == "__main__":
    # Run single example
    example_workflow()
    
    # Uncomment to simulate multiple cases
    # simulate_multiple_cases()
