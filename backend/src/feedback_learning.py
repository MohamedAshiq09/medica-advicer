"""
Feedback Learning Agent - Learns from user corrections to improve predictions
Responsibility: Collect feedback, retrain models, and improve accuracy over time
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import joblib

from .config import TRIAGE_MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackLearningAgent:
    """
    Agent that learns from user feedback to improve predictions
    """
    
    def __init__(self):
        self.feedback_file = Path("backend/data/user_feedback.jsonl")
        self.feedback_stats_file = Path("backend/data/feedback_stats.json")
        self.feedback_data = []
        self.load_feedback()
        
    def load_feedback(self):
        """Load existing feedback from file"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.feedback_data)} feedback entries")
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                self.feedback_data = []
    
    def record_feedback(self, 
                       symptoms: List[str],
                       predicted_label: str,
                       user_correction: str,
                       user_notes: str = "",
                       severity_info: Dict = None) -> bool:
        """
        Record user feedback on a prediction
        
        Args:
            symptoms: List of detected symptoms
            predicted_label: What model predicted
            user_correction: What user says is correct (self-care, see_doctor, emergency)
            user_notes: Optional notes from user
            severity_info: Optional severity information
            
        Returns:
            True if feedback recorded successfully
        """
        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "symptoms": symptoms,
                "predicted_label": predicted_label,
                "correct_label": user_correction,
                "user_notes": user_notes,
                "severity_info": severity_info or {},
                "was_correct": predicted_label == user_correction
            }
            
            self.feedback_data.append(feedback_entry)
            
            # Append to file
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_file, 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            
            logger.info(f"Feedback recorded: {user_correction} (was_correct: {feedback_entry['was_correct']})")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict:
        """
        Get statistics about feedback collected
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self.feedback_data:
            return {
                "total_feedback": 0,
                "accuracy_from_feedback": 0,
                "by_label": {}
            }
        
        df = pd.DataFrame(self.feedback_data)
        total = len(df)
        correct = df['was_correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        stats = {
            "total_feedback": total,
            "correct_predictions": int(correct),
            "accuracy_from_feedback": float(accuracy),
            "by_label": {}
        }
        
        # Stats by predicted label
        for label in ['self-care', 'see_doctor', 'emergency']:
            label_data = df[df['predicted_label'] == label]
            if len(label_data) > 0:
                label_correct = label_data['was_correct'].sum()
                stats["by_label"][label] = {
                    "total": len(label_data),
                    "correct": int(label_correct),
                    "accuracy": float(label_correct / len(label_data))
                }
        
        # Most common corrections
        if 'correct_label' in df.columns:
            corrections = df[df['was_correct'] == False]['correct_label'].value_counts().to_dict()
            stats["common_corrections"] = corrections
        
        return stats
    
    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained based on feedback
        
        Returns:
            True if retraining is recommended
        """
        if len(self.feedback_data) < 5:
            return False
        
        # Retrain if accuracy from feedback is significantly lower than model accuracy
        stats = self.get_feedback_stats()
        feedback_accuracy = stats.get('accuracy_from_feedback', 1.0)
        
        # Retrain if accuracy drops below 70%
        return feedback_accuracy < 0.70
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from feedback for retraining
        
        Returns:
            Tuple of (X, y, symptom_list)
        """
        if not self.feedback_data:
            return None, None, []
        
        df = pd.DataFrame(self.feedback_data)
        
        # Get all unique symptoms
        all_symptoms = set()
        for symptoms in df['symptoms']:
            all_symptoms.update(symptoms)
        
        symptom_list = sorted(list(all_symptoms))
        
        # Create feature matrix
        X = []
        for symptoms in df['symptoms']:
            feature_vector = [1 if symptom in symptoms else 0 for symptom in symptom_list]
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Create label vector
        label_map = {'self-care': 0, 'see_doctor': 1, 'emergency': 2}
        y = np.array([label_map[label] for label in df['correct_label']])
        
        return X, y, symptom_list
    
    def retrain_model(self) -> Dict:
        """
        Retrain model using feedback data
        
        Returns:
            Dictionary with retraining results
        """
        X, y, symptom_list = self.prepare_training_data()
        
        if X is None or len(X) < 3:
            logger.warning("Not enough feedback data to retrain")
            return {"success": False, "reason": "Insufficient data"}
        
        try:
            # Try multiple models and pick the best
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'naive_bayes': GaussianNB()
            }
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=min(3, len(X)))
                    avg_score = scores.mean()
                    
                    logger.info(f"{name} CV score: {avg_score:.3f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model_name = name
                except Exception as e:
                    logger.warning(f"Error with {name}: {e}")
            
            if best_model is None:
                return {"success": False, "reason": "All models failed"}
            
            # Train on full data
            best_model.fit(X, y)
            
            # Save the retrained model
            joblib.dump(best_model, TRIAGE_MODEL_PATH)
            logger.info(f"Model retrained with {best_model_name}, CV score: {best_score:.3f}")
            
            return {
                "success": True,
                "model_type": best_model_name,
                "cv_score": float(best_score),
                "training_samples": len(X),
                "symptoms_used": symptom_list
            }
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return {"success": False, "reason": str(e)}
    
    def get_confidence_adjustment(self, symptoms: List[str], 
                                 base_confidence: float) -> float:
        """
        Adjust confidence based on feedback history for similar symptoms
        
        Args:
            symptoms: List of symptoms
            base_confidence: Original model confidence
            
        Returns:
            Adjusted confidence score
        """
        if not self.feedback_data:
            return base_confidence
        
        # Find similar cases in feedback
        symptom_set = set(symptoms)
        similar_cases = []
        
        for feedback in self.feedback_data:
            feedback_symptoms = set(feedback['symptoms'])
            # Calculate Jaccard similarity
            if len(symptom_set | feedback_symptoms) > 0:
                similarity = len(symptom_set & feedback_symptoms) / len(symptom_set | feedback_symptoms)
                if similarity > 0.5:  # Similar enough
                    similar_cases.append({
                        'similarity': similarity,
                        'was_correct': feedback['was_correct']
                    })
        
        if not similar_cases:
            return base_confidence
        
        # Adjust based on accuracy of similar cases
        similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_cases[:5]
        
        accuracy_of_similar = sum(1 for case in top_similar if case['was_correct']) / len(top_similar)
        
        # Blend with base confidence
        adjusted = 0.7 * base_confidence + 0.3 * accuracy_of_similar
        
        return min(0.95, max(0.3, adjusted))  # Clamp between 0.3 and 0.95

# Global instance
feedback_agent = FeedbackLearningAgent()

def record_user_feedback(symptoms: List[str],
                        predicted_label: str,
                        user_correction: str,
                        user_notes: str = "",
                        severity_info: Dict = None) -> bool:
    """Record feedback from user"""
    return feedback_agent.record_feedback(symptoms, predicted_label, user_correction, user_notes, severity_info)

def get_feedback_statistics() -> Dict:
    """Get feedback statistics"""
    return feedback_agent.get_feedback_stats()

def should_retrain_model() -> bool:
    """Check if model should be retrained"""
    return feedback_agent.should_retrain()

def retrain_from_feedback() -> Dict:
    """Retrain model using feedback"""
    return feedback_agent.retrain_model()

def adjust_confidence(symptoms: List[str], base_confidence: float) -> float:
    """Adjust confidence based on feedback"""
    return feedback_agent.get_confidence_adjustment(symptoms, base_confidence)
