import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Data Preprocessing & Feature Engineering Pipeline for Student Performance Dataset.
    Handles data cleaning, domain-specific feature engineering, encoding, and scaling.
    """

    def __init__(self, target_type: str = "regression"):
        """
        Parameters:
            target_type (str): 'regression' (predicts final_exam_score) or 
                               'classification' (predicts final_grade).
        """
        self.target_type = target_type
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names: list = []
        self.grade_mapping = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
        self.inverse_grade_mapping = {v: k for k, v in self.grade_mapping.items()}

    def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataframe and creates new predictive features.
        """
        df = df.copy()

        # 1. Handle missing values in parental_education
        if 'parental_education' in df.columns:
            df['parental_education'] = df['parental_education'].fillna('None').astype(str)

        # 2. Domain Feature Engineering
        # Ratio of study time to sleep duration
        if 'study_time_hours' in df.columns and 'sleep_hours' in df.columns:
            df['study_to_sleep_ratio'] = df['study_time_hours'] / (df['sleep_hours'] + 1e-5)

        # Effort Index: interaction between study time and attendance
        if 'study_time_hours' in df.columns and 'attendance_percent' in df.columns:
            df['academic_effort_index'] = (df['study_time_hours'] * df['attendance_percent']) / 100.0

        # Academic momentum: interaction between previous grade and study time
        if 'previous_grade' in df.columns and 'study_time_hours' in df.columns:
            df['academic_momentum'] = (df['previous_grade'] * df['study_time_hours']) / 10.0

        # At-Risk Indicator (Low attendance < 75% OR study time < 2 hours)
        if 'attendance_percent' in df.columns and 'study_time_hours' in df.columns:
            df['is_at_risk'] = ((df['attendance_percent'] < 75.0) | (df['study_time_hours'] < 2.0)).astype(int)

        return df

    def build_transformer_pipeline(self) -> ColumnTransformer:
        """
        Builds a Scikit-Learn ColumnTransformer for numerical and categorical features.
        """
        # Feature column definitions
        numerical_features = [
            'study_time_hours', 
            'attendance_percent', 
            'sleep_hours', 
            'previous_grade',
            'study_to_sleep_ratio',
            'academic_effort_index',
            'academic_momentum'
        ]

        ordinal_features = ['parental_education']
        education_categories = [['None', 'High School', 'Bachelors', 'Masters', 'PhD']]

        binary_categorical_features = [
            'gender', 
            'internet_access', 
            'extracurricular_activities', 
            'part_time_job'
        ]

        passthrough_features = ['is_at_risk']

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        ord_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('encoder', OrdinalEncoder(categories=education_categories, handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numerical_features),
                ('ord', ord_pipeline, ordinal_features),
                ('cat', cat_pipeline, binary_categorical_features),
                ('pass', 'passthrough', passthrough_features)
            ],
            remainder='drop'
        )

        return preprocessor

    def prepare_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Executes full preprocessing and feature engineering, then splits into train/test sets.
        """
        # Step 1: Feature Engineering
        df_engineered = self.clean_and_engineer_features(df)

        # Step 2: Separate features and target
        drop_cols = ['student_id', 'final_exam_score', 'final_grade']
        X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])

        if self.target_type == "regression":
            y = df_engineered['final_exam_score'].values
            stratify = None
        elif self.target_type == "classification":
            y = df_engineered['final_grade'].map(self.grade_mapping).values
            stratify = y
        else:
            raise ValueError("target_type must be 'regression' or 'classification'")

        # Step 3: Train / Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Step 4: Fit transformer on training set only to prevent data leakage
        self.pipeline = self.build_transformer_pipeline()
        X_train_transformed = self.pipeline.fit_transform(X_train_raw)
        X_test_transformed = self.pipeline.transform(X_test_raw)

        # Retrieve output feature names
        try:
            self.feature_names = self.pipeline.get_feature_names_out().tolist()
        except Exception:
            self.feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

        return X_train_transformed, X_test_transformed, y_train, y_test, self.feature_names

    def save_pipeline(self, filepath: str = "models/preprocessor.joblib"):
        """
        Saves the fitted transformer pipeline for deployment inference.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"[+] Fitted preprocessor pipeline saved to: {filepath}")

    @staticmethod
    def load_pipeline(filepath: str = "models/preprocessor.joblib"):
        """
        Loads the fitted transformer pipeline.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        return joblib.load(filepath)


def run_preprocessing_pipeline(data_path="data/student_performance_dataset.csv", output_dir="data/processed"):
    """
    Executes the preprocessing workflow and saves processed train/test datasets to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Load Raw Data
    df = pd.read_csv(data_path)
    print(f"\n[+] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Regression Preprocessing
    print("\n--- [A] Building Regression Pipeline (Target: final_exam_score) ---")
    reg_preprocessor = DataPreprocessor(target_type="regression")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names = reg_preprocessor.prepare_data(df)

    print(f"    - Processed Features Count: {len(feature_names)}")
    print(f"    - Feature Names: {feature_names}")
    print(f"    - Train Set Shape: X={X_train_reg.shape}, y={y_train_reg.shape}")
    print(f"    - Test Set Shape : X={X_test_reg.shape}, y={y_test_reg.shape}")

    # Save fitted preprocessor
    reg_preprocessor.save_pipeline("models/preprocessor.joblib")

    # Save processed arrays as DataFrames for inspection / training
    train_reg_df = pd.DataFrame(X_train_reg, columns=feature_names)
    train_reg_df['final_exam_score'] = y_train_reg
    train_reg_df.to_csv(os.path.join(output_dir, "train_regression.csv"), index=False)

    test_reg_df = pd.DataFrame(X_test_reg, columns=feature_names)
    test_reg_df['final_exam_score'] = y_test_reg
    test_reg_df.to_csv(os.path.join(output_dir, "test_regression.csv"), index=False)

    # 3. Classification Preprocessing
    print("\n--- [B] Building Classification Pipeline (Target: final_grade) ---")
    clf_preprocessor = DataPreprocessor(target_type="classification")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, _ = clf_preprocessor.prepare_data(df)

    print(f"    - Train Set Shape: X={X_train_clf.shape}, y={y_train_clf.shape}")
    print(f"    - Test Set Shape : X={X_test_clf.shape}, y={y_test_clf.shape}")

    train_clf_df = pd.DataFrame(X_train_clf, columns=feature_names)
    train_clf_df['final_grade'] = y_train_clf
    train_clf_df.to_csv(os.path.join(output_dir, "train_classification.csv"), index=False)

    test_clf_df = pd.DataFrame(X_test_clf, columns=feature_names)
    test_clf_df['final_grade'] = y_test_clf
    test_clf_df.to_csv(os.path.join(output_dir, "test_classification.csv"), index=False)

    print(f"\n[+] All processed datasets saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    run_preprocessing_pipeline()
