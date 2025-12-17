"""
Test script to verify the feedback learning system works correctly
"""
import sys
sys.path.insert(0, '/backend')

from src.feedback_learning import (
    record_user_feedback,
    get_feedback_statistics,
    should_retrain_model,
    adjust_confidence
)

def test_feedback_recording():
    """Test recording feedback"""
    print("\n" + "="*60)
    print("TEST 1: Recording Feedback")
    print("="*60)
    
    success = record_user_feedback(
        symptoms=["fever", "headache"],
        predicted_label="see_doctor",
        user_correction="self-care",
        user_notes="Mild fever, managed at home"
    )
    
    print(f"✓ Feedback recorded: {success}")
    assert success, "Failed to record feedback"

def test_feedback_statistics():
    """Test getting feedback statistics"""
    print("\n" + "="*60)
    print("TEST 2: Feedback Statistics")
    print("="*60)
    
    stats = get_feedback_statistics()
    
    print(f"✓ Total feedback: {stats['total_feedback']}")
    print(f"✓ Accuracy: {stats['accuracy_from_feedback']:.2%}")
    print(f"✓ By label: {stats['by_label']}")
    
    assert 'total_feedback' in stats, "Missing total_feedback"
    assert 'accuracy_from_feedback' in stats, "Missing accuracy_from_feedback"

def test_confidence_adjustment():
    """Test confidence adjustment"""
    print("\n" + "="*60)
    print("TEST 3: Confidence Adjustment")
    print("="*60)
    
    symptoms = ["fever", "headache"]
    base_confidence = 0.95
    
    adjusted = adjust_confidence(symptoms, base_confidence)
    
    print(f"✓ Base confidence: {base_confidence:.2%}")
    print(f"✓ Adjusted confidence: {adjusted:.2%}")
    
    assert 0.0 <= adjusted <= 1.0, "Confidence out of range"

def test_retrain_check():
    """Test retrain check"""
    print("\n" + "="*60)
    print("TEST 4: Retrain Check")
    print("="*60)
    
    needs_retrain = should_retrain_model()
    
    print(f"✓ Retrain needed: {needs_retrain}")
    print(f"  (Retraining triggers when accuracy < 70%)")

def test_multiple_feedback():
    """Test recording multiple feedback entries"""
    print("\n" + "="*60)
    print("TEST 5: Multiple Feedback Entries")
    print("="*60)
    
    test_cases = [
        {
            "symptoms": ["cough", "runny_nose"],
            "predicted": "self-care",
            "correct": "self-care",
            "notes": "Common cold"
        },
        {
            "symptoms": ["chest_pain", "shortness_breath"],
            "predicted": "see_doctor",
            "correct": "emergency",
            "notes": "Severe symptoms"
        },
        {
            "symptoms": ["headache", "fever"],
            "predicted": "see_doctor",
            "correct": "self-care",
            "notes": "Mild symptoms"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        success = record_user_feedback(
            symptoms=case['symptoms'],
            predicted_label=case['predicted'],
            user_correction=case['correct'],
            user_notes=case['notes']
        )
        print(f"✓ Case {i}: {case['notes']} - Recorded: {success}")
    
    stats = get_feedback_statistics()
    print(f"\n✓ Total feedback entries: {stats['total_feedback']}")
    print(f"✓ Overall accuracy: {stats['accuracy_from_feedback']:.2%}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("FEEDBACK LEARNING SYSTEM - TEST SUITE")
    print("="*60)
    
    try:
        test_feedback_recording()
        test_feedback_statistics()
        test_confidence_adjustment()
        test_retrain_check()
        test_multiple_feedback()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe feedback learning system is working correctly.")
        print("You can now:")
        print("1. Record user feedback via /feedback/record endpoint")
        print("2. Check statistics via /feedback/stats endpoint")
        print("3. Trigger retraining via /model/retrain endpoint")
        print("\nSee FEEDBACK_LEARNING_GUIDE.md for detailed documentation.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
