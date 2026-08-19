# Student Academic Performance Analytics & AI Predictor

An end-to-end Data Science and Machine Learning project exploring the behavioral, socio-demographic, and academic determinants of student achievement (final_exam_score and final_grade).

Includes automated exploratory data analysis, feature engineering pipelines, multi-model benchmarks, hyperparameter tuning, model explainability, and an interactive Streamlit Web Application.

---

## Project Architecture

```text
Student Performance Project/
│
├── data/
│   ├── student_performance_dataset.csv     # Raw dataset (1,000 students)
│   └── processed/                          # Scaled, encoded train/test datasets
│       ├── train_regression.csv            # 800 training rows (13 engineered features)
│       ├── test_regression.csv             # 200 holdout testing rows
│       ├── train_classification.csv
│       └── test_classification.csv
│
├── notebooks/                              # Interactive Jupyter Notebooks
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis & Visualizations
│   ├── 02_preprocessing.ipynb              # Feature Pipelines & Transformation
│   └── 03_modeling.ipynb                   # Model Training, Tuning & Evaluation
│
├── src/                                    # Modular Production-Ready Code
│   ├── __init__.py
│   ├── data_loader.py                      # Data ingestion & path resolver
│   ├── eda_analysis.py                     # Automated statistical analysis & figure generation
│   ├── preprocess.py                       # Data cleaning, encoding, scaling & pipeline export
│   ├── train.py                            # Model training, cross-validation & hyperparameter tuning
│   └── predict.py                          # Real-time inference & recommendation engine
│
├── models/                                 # Serialized Production Artifacts
│   ├── preprocessor.joblib                 # Fitted Scikit-Learn ColumnTransformer
│   ├── best_regression_model.joblib        # Top performing score regressor (Lasso / Ridge)
│   └── best_classification_model.joblib    # Top performing grade classifier (Gradient Boosting)
│
├── reports/
│   ├── model_evaluation_metrics.json       # Consolidated benchmark metrics
│   └── figures/                            # High-resolution presentation charts (01-09)
│       ├── 01_target_distributions.png
│       ├── 02_numeric_distributions.png
│       ├── 03_correlation_heatmap.png
│       ├── 04_key_relationships_regression.png
│       ├── 05_categorical_impact_boxplots.png
│       ├── 06_model_comparison_regression.png
│       ├── 07_actual_vs_predicted_residuals.png
│       ├── 08_feature_importance.png
│       └── 09_confusion_matrix_classification.png
│
├── app/
│   └── app.py                              # Interactive Streamlit Web Application
│
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## Project Phases & Lifecycle

### Step 1: Exploratory Data Analysis (EDA)
* **Dataset Characteristics:** 1,000 students, 12 initial features, 0 duplicates.
* **Correlations with Exam Score:**
  * **Study Time (study_time_hours):** r = +0.568 (Strongest single positive driver).
  * **Previous Academic Grade (previous_grade):** r = +0.406.
  * **Attendance Rate (attendance_percent):** r = +0.262.
* **Socio-Demographic Disparities:**
  * Students with home internet access averaged 84.28 vs 79.26 without internet (~5.0 pt gap).
  * Students with part-time jobs averaged 81.29 vs 84.58 without jobs (~3.3 pt gap).

### Step 2: Feature Engineering & Preprocessing Pipeline
* **Domain Feature Creation:**
  * `study_to_sleep_ratio`: Balances study workload against rest.
  * `academic_effort_index`: Interaction metric between study hours and attendance rate.
  * `academic_momentum`: Interaction between previous grades and study consistency.
  * `is_at_risk`: Binary indicator flag for students with < 75% attendance or < 2.0 daily study hours.
* **Transformations:** `StandardScaler` for continuous features, `OrdinalEncoder` for parental education levels, and `OneHotEncoder` for nominal attributes.

### Step 3: Machine Learning Modeling & Benchmarking
* **Regression Benchmark (Predicting final_exam_score):**
  * **Lasso Regression (Tuned):** Test MAE = 4.933, RMSE = 6.367, R2 = 0.6233
  * **Ridge Regression:** Test MAE = 5.015, RMSE = 6.461, R2 = 0.6121
  * **Linear Regression:** Test MAE = 5.030, RMSE = 6.472, R2 = 0.6108
* **Classification Benchmark (Predicting Grade A, B, C, D, F):**
  * **Gradient Boosting Classifier:** Test Accuracy = 56.00%, Weighted F1 = 0.5545
  * **Random Forest Classifier:** Test Accuracy = 55.50%, Weighted F1 = 0.5562

### Step 4: Interactive Web Application (Streamlit)
* **Tab 1: Single Student Simulator:** Adjust sliders (study hours, sleep, attendance, demographics) for instant predicted scores, grade breakdown, risk alerts, and tailored academic tips.
* **Tab 2: Interactive EDA Explorer:** Real-time cohort slicing by grade, employment status, and internet access.
* **Tab 3: Model Leaderboard & Explainability:** Full evaluation table, feature importances, and confusion matrix.
* **Tab 4: Batch CSV Scoring:** Upload a roster of students to score them automatically and export an enriched CSV.

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Step 1 (EDA)
```bash
python src/eda_analysis.py
```

### 3. Run Step 2 (Preprocessing)
```bash
python src/preprocess.py
```

### 4. Run Step 3 (Model Training & Evaluation)
```bash
python src/train.py
```

### 5. Launch Step 4 (Streamlit Web Dashboard)
```bash
streamlit run app/app.py
```
