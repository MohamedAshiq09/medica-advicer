#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

print("Testing imports...")

try:
    print("1. Testing FastAPI imports...")
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict
    import logging
    from datetime import datetime
    print("   ✓ FastAPI imports successful")
except Exception as e:
    print(f"   ✗ FastAPI imports failed: {e}")
    sys.exit(1)

try:
    print("2. Testing config import...")
    from src.config import SYMPTOM_VOCABULARY, TRIAGE_LEVELS
    print("   ✓ Config import successful")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")

try:
    print("3. Testing preprocessing import...")
    from src.preprocessing import preprocess_input
    print("   ✓ Preprocessing import successful")
except Exception as e:
    print(f"   ✗ Preprocessing import failed: {e}")

try:
    print("4. Testing model inference import...")
    from src.model_inference import load_model, predict_triage, get_model_status
    print("   ✓ Model inference import successful")
except Exception as e:
    print(f"   ✗ Model inference import failed: {e}")

try:
    print("5. Testing apriori rules import...")
    from src.rules_apriori import load_apriori_rules, apply_apriori_rules
    print("   ✓ Apriori rules import successful")
except Exception as e:
    print(f"   ✗ Apriori rules import failed: {e}")

try:
    print("6. Testing safety layer import...")
    from src.safety_layer import apply_safety_overrides, generate_safety_message
    print("   ✓ Safety layer import successful")
except Exception as e:
    print(f"   ✗ Safety layer import failed: {e}")

try:
    print("7. Testing LLM layer import...")
    from src.llm_layer import generate_explanation, generate_follow_up_questions, is_llm_available
    print("   ✓ LLM layer import successful")
except Exception as e:
    print(f"   ✗ LLM layer import failed: {e}")

print("\n✅ All imports tested!")
print("Now testing a simple preprocessing call...")

try:
    features, symptoms = preprocess_input("I have a headache and fever")
    print(f"   ✓ Preprocessing test successful: {symptoms}")
except Exception as e:
    print(f"   ✗ Preprocessing test failed: {e}")

print("\n🎉 Import test completed!")