#!/usr/bin/env python3
"""
Final Model Training Script
Trains a production-ready model using the full processed dataset
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from src.config import MODELS_DIR, TRIAGE_LEVELS

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

def load_training_data():
    """Load the training dataset"""
    print("📊 Loading training data...")
    
    # Use expanded dataset (has more samples)
    df = pd.read_csv('data/expanded_symptom_cases.csv')
    print(f"   ✅ Loaded expanded dataset with {len(df)} samples")
    return df

def prepare_features(df):
    """Prepare features from expanded dataset"""
    print("🔧 Preparing features...")
    
    X_text = df['complaint_text']
    y = df['triage_label']
    
    # Create basic features
    X_features = pd.DataFrame()
    X_features['age'] = df['age'].fillna(40)
    X_features['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    print(f"   Text samples: {len(X_text)}")
    print(f"   Target distribution:")
    for label in sorted(y.unique()):
        count = (y == label).sum()
        label_name = TRIAGE_LEVELS.get(label, f"unknown_{label}")
        print(f"     {label_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X_text, X_features, y

def train_hybrid_model(X_text, X_features, y):
    """Train a hybrid model combining text and features"""
    print("🤖 Training hybrid model...")
    
    # Split data
    X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF vectorization
    print("   Vectorizing text...")
    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # Scale numeric features
    print("   Scaling numeric features...")
    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, X_train_features_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_features_scaled])
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train
        model.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"     Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        print(f"     CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Print classification report
        print(f"     Classification Report:")
        report = classification_report(y_test, y_pred, 
                                      target_names=[TRIAGE_LEVELS.get(i, f"class_{i}") for i in sorted(y.unique())],
                                      zero_division=0)
        for line in report.split('\n'):
            if line.strip():
                print(f"       {line}")
    
    return results, tfidf, scaler, X_test_combined, y_test

def select_and_save_best_model(results, tfidf, scaler):
    """Select and save the best model"""
    print("🏆 Selecting best model...")
    
    # Find best by F1 score (better for imbalanced data)
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_result = results[best_name]
    best_model = best_result['model']
    
    print(f"   Best model: {best_name}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"   F1 Score: {best_result['f1']:.3f}")
    print(f"   CV Score: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
    
    # Save model
    model_path = MODELS_DIR / "triage_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf, vectorizer_path)
    print(f"   ✅ TF-IDF vectorizer saved to: {vectorizer_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Feature scaler saved to: {scaler_path}")
    
    # Save label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(TRIAGE_LEVELS.values()))
    label_encoder_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(label_encoder, label_encoder_path)
    print(f"   ✅ Label encoder saved to: {label_encoder_path}")
    
    return best_name, best_result

def test_model_on_examples():
    """Test the trained model on critical examples"""
    print("🧪 Testing model on critical examples...")
    
    # Load the saved model and vectorizer
    model = joblib.load(MODELS_DIR / "triage_model.joblib")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")
    
    test_cases = [
        ("I have a mild headache", "self-care"),
        ("Severe chest pain and can't breathe", "emergency"),
        ("My brain feels like it's missing, severe head pain", "emergency"),
        ("Worst headache of my life with confusion", "emergency"),
        ("Runny nose and sore throat", "self-care"),
        ("High fever with severe headache", "see_doctor"),
        ("Crushing chest pain radiating to arm", "emergency"),
        ("Head trauma with severe pain", "emergency"),
        ("Mild cough and runny nose", "self-care"),
        ("Difficulty breathing with chest tightness", "emergency"),
    ]
    
    print("   Test Results:")
    correct = 0
    for text, expected in test_cases:
        # Vectorize text
        text_vec = tfidf.transform([text]).toarray()
        
        # Create dummy features (age=40, gender=M)
        dummy_features = np.array([[40, 1]])
        dummy_features_scaled = scaler.transform(dummy_features)
        
        # Combine - convert to dense array
        X = np.hstack([text_vec, dummy_features_scaled])
        
        # Predict
        prediction_idx = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        predicted_label = TRIAGE_LEVELS[prediction_idx]
        confidence = max(probabilities)
        
        is_correct = predicted_label == expected
        correct += is_correct
        status = "✅" if is_correct else "❌"
        
        print(f"     {status} '{text[:40]}...'")
        print(f"        Expected: {expected}, Got: {predicted_label} ({confidence:.2f})")
    
    accuracy = correct / len(test_cases) * 100
    print(f"\n   Test Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

def main():
    """Main training pipeline"""
    print("🏥 Final Medical Model Training")
    print("=" * 60)
    
    # Load data
    df = load_training_data()
    
    # Prepare features
    X_text, X_features, y = prepare_features(df)
    
    # Train hybrid model
    results, tfidf, scaler, X_test, y_test = train_hybrid_model(X_text, X_features, y)
    
    # Select and save best model
    best_name, best_result = select_and_save_best_model(results, tfidf, scaler)
    
    # Test model
    test_model_on_examples()
    
    print("\n" + "=" * 60)
    print("🎉 Model training completed successfully!")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_result['accuracy']:.1%}")
    print(f"F1 Score: {best_result['f1']:.1%}")
    print("\nThe model is now ready for production use!")
    print("=" * 60)

if __name__ == "__main__":
    main()
