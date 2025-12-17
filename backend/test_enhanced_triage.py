"""
Test the enhanced triage engine with medical knowledge base
"""
import sys
sys.path.insert(0, '/backend')

from src.enhanced_triage import perform_triage
from src.medical_knowledge_base import get_triage_recommendation, check_emergency

def test_case(description: str, symptoms: list, expected_level: str):
    """Test a single case"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Symptoms: {symptoms}")
    
    result = perform_triage(symptoms)
    
    print(f"\nResult:")
    print(f"  Triage Level: {result['triage_level']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Severity: {result.get('severity', 'Unknown')}")
    print(f"  Reason: {result['reason']}")
    print(f"  Advice: {result['advice']}")
    if result.get('matched_condition'):
        print(f"  Matched Condition: {result['matched_condition']}")
    if result.get('red_flags'):
        print(f"  Red Flags: {result['red_flags']}")
    
    # Check if result matches expected
    if result['triage_level'] == expected_level:
        print(f"\n[PASS] Got expected level: {expected_level}")
        return True
    else:
        print(f"\n[FAIL] Expected {expected_level}, got {result['triage_level']}")
        return False

def run_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print("ENHANCED TRIAGE ENGINE - TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Mild fever + headache (should be self-care)
    results.append(test_case(
        "Mild fever and headache",
        ["fever", "headache"],
        "self-care"
    ))
    
    # Test 2: Severe headache + fever + stiff neck (should be emergency)
    results.append(test_case(
        "Severe headache + fever + stiff neck (possible meningitis)",
        ["fever", "severe_headache", "stiff_neck"],
        "emergency"
    ))
    
    # Test 3: Chest pain (should be emergency)
    results.append(test_case(
        "Chest pain",
        ["chest_pain"],
        "emergency"
    ))
    
    # Test 4: Chest pain + shortness of breath (should be emergency)
    results.append(test_case(
        "Chest pain + shortness of breath",
        ["chest_pain", "shortness_breath"],
        "emergency"
    ))
    
    # Test 5: Common cold symptoms (should be self-care)
    results.append(test_case(
        "Common cold symptoms",
        ["runny_nose", "cough", "sore_throat"],
        "self-care"
    ))
    
    # Test 6: Sore throat + fever + swollen lymph nodes (should be see_doctor)
    results.append(test_case(
        "Possible strep throat",
        ["sore_throat", "fever", "swollen_lymph_nodes"],
        "see_doctor"
    ))
    
    # Test 7: Mild cough (should be self-care)
    results.append(test_case(
        "Mild cough",
        ["cough"],
        "self-care"
    ))
    
    # Test 8: Persistent cough (should be see_doctor)
    results.append(test_case(
        "Persistent cough",
        ["persistent_cough"],
        "see_doctor"
    ))
    
    # Test 9: Nausea + vomiting + diarrhea (should be self-care)
    results.append(test_case(
        "Gastroenteritis symptoms",
        ["nausea", "vomiting", "diarrhea"],
        "self-care"
    ))
    
    # Test 10: Severe abdominal pain (should be see_doctor)
    results.append(test_case(
        "Severe abdominal pain",
        ["severe_abdominal_pain"],
        "see_doctor"
    ))
    
    # Test 11: Shortness of breath (should be emergency)
    results.append(test_case(
        "Shortness of breath",
        ["shortness_breath"],
        "emergency"
    ))
    
    # Test 12: Mild fatigue (should be self-care)
    results.append(test_case(
        "Mild fatigue",
        ["fatigue"],
        "self-care"
    ))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        return True
    else:
        print(f"\n[FAILED] {total - passed} TEST(S) FAILED")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
