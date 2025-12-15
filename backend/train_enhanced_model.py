#!/usr/bin/env python3
"""
Enhanced Model Training Script
Trains a high-accuracy model for medical symptom triage
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from src.config import MODELS_DIR, TRIAGE_LEVELS
from src.preprocessing import PreprocessingAgent

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

def load_and_prepare_data():
    """Load and prepare enhanced training data"""
    print("📊 Loading enhanced training data...")
    
    # Load enhanced dataset
    df = pd.read_csv('data/enhanced_symptom_cases.csv')
    print(f"   Loaded {len(df)} training samples")
    
    # Check distribution
    print(f"   Triage distribution:")
    for label, name in TRIAGE_LEVELS.items():
        count = (df['triage_label'] == label).sum()
        print(f"     {name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    return df

def create_enhanced_features(df):
    """Create enhanced features using preprocessing agent"""
    print("🔧 Creating enhanced features...")
    
    preprocessor = PreprocessingAgent()
    
    # Process all text data
    processed_data = []
    for _, row in df.iterrows():
        features, _ = preprocessor.create_feature_vector(
            row['complaint_text'], 
            row.get('age'), 
            row.get('gender')
        )
        processed_data.append(features)
    
    processed_df = pd.DataFrame(processed_data)
    processed_df['triage_label'] = df['triage_label']
    
    print(f"   Created {processed_df.shape[1]-1} features")
    return processed_df

def train_text_based_models(df):
    """Train text-based models with enhanced TF-IDF"""
    print("🤖 Training text-based models...")
    
    X_text = df['complaint_text']
    y = df['triage_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Enhanced TF-IDF with medical-specific parameters
    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for medical phrases
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,  # Better for medical text
        lowercase=True
    )
    
    # Fit TF-IDF
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            class_weight='balanced',  # Handle class imbalance
            C=1.0
        ),
        'SVM': SVC(
            random_state=42,
            probability=True,
            class_weight='balanced',
            kernel='linear',
            C=1.0
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results, tfidf, X_test, y_test

def train_feature_based_models(processed_df):
    """Train feature-based models"""
    print("🧠 Training feature-based models...")
    
    # Prepare features - exclude text columns
    exclude_columns = ['triage_label', 'original_text', 'cleaned_text']
    feature_columns = [col for col in processed_df.columns if col not in exclude_columns]
    
    # Handle categorical columns
    X = processed_df[feature_columns].copy()
    
    # Convert categorical columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            # Simple label encoding for categorical variables
            unique_vals = X[col].unique()
            val_map = {val: i for i, val in enumerate(unique_vals)}
            X[col] = X[col].map(val_map)
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    y = processed_df['triage_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Enhanced Logistic': LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            C=0.1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results

def select_and_save_best_model(text_results, feature_results, tfidf):
    """Select and save the best performing model"""
    print("🏆 Selecting best model...")
    
    # Combine all results
    all_results = {**text_results, **feature_results}
    
    # Find best model by accuracy
    best_name = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    best_result = all_results[best_name]
    best_model = best_result['model']
    
    print(f"   Best model: {best_name}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"   CV Score: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
    
    # Save model
    model_path = MODELS_DIR / "triage_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"   Model saved to: {model_path}")
    
    # Save TF-IDF if it's a text-based model
    if best_name in text_results:
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
        joblib.dump(tfidf, vectorizer_path)
        print(f"   TF-IDF vectorizer saved to: {vectorizer_path}")
    
    # Save label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(TRIAGE_LEVELS.values()))
    label_encoder_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(label_encoder, label_encoder_path)
    print(f"   Label encoder saved to: {label_encoder_path}")
    
    return best_name, best_result

def test_model_on_examples():
    """Test the trained model on specific examples"""
    print("🧪 Testing model on critical examples...")
    
    # Load the saved model and vectorizer
    model = joblib.load(MODELS_DIR / "triage_model.joblib")
    
    try:
        tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        is_text_model = True
    except:
        is_text_model = False
    
    test_cases = [
        ("I have a mild headache", "self-care"),
        ("Severe chest pain and can't breathe", "emergency"),
        ("My brain feels like it's missing, severe head pain", "emergency"),
        ("Worst headache of my life with confusion", "emergency"),
        ("Runny nose and sore throat", "self-care"),
        ("High fever with severe headache", "see_doctor"),
        ("Crushing chest pain radiating to arm", "emergency"),
        ("Head trauma with severe pain", "emergency")
    ]
    
    print("   Test Results:")
    for text, expected in test_cases:
        if is_text_model:
            # Text-based prediction
            text_vector = tfidf.transform([text])
            prediction_idx = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
        else:
            # Feature-based prediction (would need preprocessing)
            # For now, skip feature-based testing
            continue
            
        predicted_label = TRIAGE_LEVELS[prediction_idx]
        confidence = max(probabilities)
        
        status = "✅" if predicted_label == expected else "❌"
        print(f"     {status} '{text[:40]}...'")
        print(f"        Expected: {expected}, Got: {predicted_label} ({confidence:.2f})")

def main():
    """Main training pipeline"""
    print("🏥 Enhanced Medical Model Training")
    print("=" * 50)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create enhanced features
    processed_df = create_enhanced_features(df)
    
    # Train text-based models
    text_results, tfidf, X_test, y_test = train_text_based_models(df)
    
    # Train feature-based models
    feature_results = train_feature_based_models(processed_df)
    
    # Select and save best model
    best_name, best_result = select_and_save_best_model(text_results, feature_results, tfidf)
    
    # Test model
    test_model_on_examples()
    
    print("\n🎉 Enhanced model training completed!")
    print(f"Best model: {best_name} with {best_result['accuracy']:.1%} accuracy")
    print("The model should now handle critical symptoms much better!")

if __name__ == "__main__":
    main()