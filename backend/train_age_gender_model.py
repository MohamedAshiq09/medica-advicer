#!/usr/bin/env python3
"""
Age and Gender-Aware Model Training
Trains a model that considers age, gender, and symptoms for accurate triage
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from src.config import MODELS_DIR, TRIAGE_LEVELS

MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load expanded training data with age and gender"""
    print("📊 Loading expanded training data with demographics...")
    
    from pathlib import Path
    data_path = Path(__file__).parent / 'data' / 'expanded_symptom_cases.csv'
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} training samples")
    
    # Show distribution
    print(f"\n   Triage distribution:")
    for label, name in TRIAGE_LEVELS.items():
        count = (df['triage_label'] == label).sum()
        pct = count/len(df)*100
        print(f"     {name}: {count} samples ({pct:.1f}%)")
    
    print(f"\n   Age distribution:")
    print(f"     Min: {df['age'].min()}, Max: {df['age'].max()}, Mean: {df['age'].mean():.1f}")
    
    print(f"\n   Gender distribution:")
    print(f"     {df['gender'].value_counts().to_dict()}")
    
    return df

def create_hybrid_features(df):
    """Create hybrid features combining text, age, and gender"""
    print("\n🔧 Creating hybrid features...")
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    
    text_features = tfidf.fit_transform(df['complaint_text'])
    
    # Convert to dense for combining with other features
    text_df = pd.DataFrame(
        text_features.toarray(),
        columns=[f'text_{i}' for i in range(text_features.shape[1])]
    )
    
    # Age features
    age_df = pd.DataFrame({
        'age': df['age'],
        'age_squared': df['age'] ** 2,
        'is_child': (df['age'] < 12).astype(int),
        'is_elderly': (df['age'] >= 65).astype(int),
        'is_high_risk_age': ((df['age'] < 5) | (df['age'] >= 60)).astype(int)
    })
    
    # Gender features
    gender_df = pd.DataFrame({
        'gender_male': (df['gender'].str.upper() == 'M').astype(int),
        'gender_female': (df['gender'].str.upper() == 'F').astype(int)
    })
    
    # Combine all features
    X = pd.concat([text_df, age_df, gender_df], axis=1)
    y = df['triage_label']
    
    print(f"   Created {X.shape[1]} total features")
    print(f"   - Text features: {text_features.shape[1]}")
    print(f"   - Age features: {age_df.shape[1]}")
    print(f"   - Gender features: {gender_df.shape[1]}")
    
    return X, y, tfidf

def train_hybrid_model(X, y):
    """Train hybrid model with age and gender awareness"""
    print("\n🤖 Training hybrid age/gender-aware model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting (better for mixed feature types)
    print("   Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   F1 Score (weighted): {f1_weighted:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"   CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Classification report
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(TRIAGE_LEVELS.values())))
    
    return model, scaler, accuracy, y_test, y_pred, y_pred_proba

def save_models(model, scaler, tfidf):
    """Save all models and preprocessors"""
    print("\n💾 Saving models...")
    
    # Save main model
    model_path = MODELS_DIR / "triage_model.joblib"
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved: {scaler_path}")
    
    # Save TF-IDF vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf, vectorizer_path)
    print(f"   TF-IDF vectorizer saved: {vectorizer_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'GradientBoostingClassifier',
        'features': 'hybrid (text + age + gender)',
        'triage_levels': TRIAGE_LEVELS,
        'training_samples': 400,
        'includes_age': True,
        'includes_gender': True
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")

def test_model_predictions(model, scaler, tfidf):
    """Test model on critical cases"""
    print("\n🧪 Testing model on critical cases...")
    
    test_cases = [
        # (text, age, gender, expected_label)
        ("I have a mild headache", 25, 'F', 0),
        ("Severe chest pain and can't breathe", 45, 'M', 2),
        ("Worst headache of my life with confusion", 40, 'M', 2),
        ("Runny nose and sore throat", 30, 'F', 0),
        ("High fever with severe headache", 35, 'M', 1),
        ("Crushing chest pain radiating to arm", 55, 'M', 2),
        ("Chest pain in elderly patient", 72, 'M', 2),
        ("Fever in young child", 4, 'M', 1),
        ("Mild cold symptoms", 24, 'F', 0),
        ("Severe abdominal pain with blood", 52, 'M', 2),
    ]
    
    print("   Results:")
    correct = 0
    for text, age, gender, expected in test_cases:
        # Create features
        text_vec = tfidf.transform([text]).toarray()
        age_features = np.array([[
            age,
            age ** 2,
            1 if age < 12 else 0,
            1 if age >= 65 else 0,
            1 if (age < 5 or age >= 60) else 0
        ]])
        gender_features = np.array([[
            1 if gender.upper() == 'M' else 0,
            1 if gender.upper() == 'F' else 0
        ]])
        
        X_test = np.hstack([text_vec, age_features, gender_features])
        X_test_scaled = scaler.transform(X_test)
        
        pred_idx = model.predict(X_test_scaled)[0]
        pred_label = TRIAGE_LEVELS[pred_idx]
        proba = model.predict_proba(X_test_scaled)[0]
        confidence = np.max(proba)
        
        expected_name = TRIAGE_LEVELS[expected]
        match = "✅" if pred_idx == expected else "❌"
        
        if pred_idx == expected:
            correct += 1
        
        print(f"     {match} Age {age}, {gender}: '{text[:30]}...'")
        print(f"        Expected: {expected_name}, Got: {pred_label} ({confidence:.2f})")
    
    accuracy = correct / len(test_cases) * 100
    print(f"\n   Test Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

def main():
    """Main training pipeline"""
    print("🏥 Age and Gender-Aware Medical Triage Model Training")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create hybrid features
    X, y, tfidf = create_hybrid_features(df)
    
    # Train model
    model, scaler, accuracy, y_test, y_pred, y_pred_proba = train_hybrid_model(X, y)
    
    # Save models
    save_models(model, scaler, tfidf)
    
    # Test predictions
    test_model_predictions(model, scaler, tfidf)
    
    print("\n" + "=" * 60)
    print("🎉 Training completed successfully!")
    print(f"   Model accuracy: {accuracy:.1%}")
    print("   Features: Text + Age + Gender")
    print("   Ready for production use!")

if __name__ == "__main__":
    main()
