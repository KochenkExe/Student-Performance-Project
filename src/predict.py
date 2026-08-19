import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.preprocess import DataPreprocessor
except ImportError:
    from preprocess import DataPreprocessor

class StudentPerformancePredictor:
    """
    Inference predictor that consumes raw student features,
    applies fitted preprocessing transformations, and outputs predicted scores & grades.
    """

    def __init__(
        self,
        preprocessor_path: str = "models/preprocessor.joblib",
        reg_model_path: str = "models/best_regression_model.joblib",
        clf_model_path: str = "models/best_classification_model.joblib"
    ):
        self.preprocessor_pipeline = joblib.load(preprocessor_path)
        self.reg_model = joblib.load(reg_model_path) if os.path.exists(reg_model_path) else None
        self.clf_model = joblib.load(clf_model_path) if os.path.exists(clf_model_path) else None
        
        self.preprocessor_helper = DataPreprocessor()
        self.grade_labels = ['F', 'D', 'C', 'B', 'A']

    def predict_single(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts final exam score and letter grade for an individual student.
        """
        df_input = pd.DataFrame([student_data])
        
        # 1. Feature Engineering
        df_engineered = self.preprocessor_helper.clean_and_engineer_features(df_input)
        
        # 2. Transform Features
        drop_cols = ['student_id', 'final_exam_score', 'final_grade']
        X_raw = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])
        X_transformed = self.preprocessor_pipeline.transform(X_raw)

        # 3. Predict Continuous Exam Score
        predicted_score = float(np.clip(self.reg_model.predict(X_transformed)[0], 0.0, 100.0))

        # 4. Predict Letter Grade
        if self.clf_model is not None:
            predicted_class_idx = int(self.clf_model.predict(X_transformed)[0])
            predicted_grade = self.grade_labels[predicted_class_idx]
            grade_probs = self.clf_model.predict_proba(X_transformed)[0]
            prob_dict = {self.grade_labels[i]: round(float(p), 4) for i, p in enumerate(grade_probs) if i < len(self.grade_labels)}
        else:
            # Rule-based fallback if classifier not loaded
            if predicted_score >= 90: predicted_grade = 'A'
            elif predicted_score >= 80: predicted_grade = 'B'
            elif predicted_score >= 70: predicted_grade = 'C'
            elif predicted_score >= 60: predicted_grade = 'D'
            else: predicted_grade = 'F'
            prob_dict = {}

        # 5. Generate Tailored Recommendations
        recommendations = []
        if student_data.get('study_time_hours', 0) < 3.0:
            recommendations.append("Increase daily focused study time to at least 3.5 - 4.5 hours.")
        if student_data.get('attendance_percent', 100) < 85.0:
            recommendations.append("Improve attendance above 90% to avoid falling behind on core lecture material.")
        if student_data.get('sleep_hours', 8) < 6.0:
            recommendations.append("Ensure 7+ hours of sleep per night to maintain cognitive endurance.")
        if student_data.get('internet_access', 'Yes') == 'No':
            recommendations.append("Utilize campus/library digital learning resources to bridge connectivity gaps.")
        if student_data.get('part_time_job', 'No') == 'Yes':
            recommendations.append("Consider structured time-blocking to balance part-time work shifts with study intervals.")

        if not recommendations:
            recommendations.append("Outstanding academic routine! Maintain current study habits and sleep schedule.")

        return {
            "predicted_final_score": round(predicted_score, 2),
            "predicted_letter_grade": predicted_grade,
            "grade_probabilities": prob_dict,
            "is_at_risk": bool(df_engineered['is_at_risk'].iloc[0]),
            "actionable_recommendations": recommendations
        }

if __name__ == "__main__":
    predictor = StudentPerformancePredictor()
    
    # Test sample profile
    sample_student = {
        "gender": "Male",
        "study_time_hours": 4.5,
        "attendance_percent": 92.0,
        "sleep_hours": 7.0,
        "parental_education": "Bachelors",
        "internet_access": "Yes",
        "extracurricular_activities": "Yes",
        "part_time_job": "No",
        "previous_grade": 82.0
    }
    
    print("\n--- Test Prediction ---")
    result = predictor.predict_single(sample_student)
    print(json.dumps(result, indent=2))
