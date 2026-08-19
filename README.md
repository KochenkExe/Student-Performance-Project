# Student Academic Performance Analytics & AI Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Pipeline-Production--Ready-success.svg)](#)

An end-to-end Data Science and Machine Learning platform designed to analyze, explain, and predict student academic achievement. The system models behavioural habits, socio-demographic determinants, and historical performance to forecast continuous exam outcomes (`final_exam_score`) and categorical letter grades (`final_grade`), while proactively flagging at-risk students and delivering actionable interventions.

---

## Table of Contents

- [Project Overview & Key Objectives](#project-overview--key-objectives)
- [Key Insights & Statistical Findings](#key-insights--statistical-findings)
- [Dataset Schema & Architecture](#dataset-schema--architecture)
- [Feature Engineering & Preprocessing Pipeline](#feature-engineering--preprocessing-pipeline)
- [Machine Learning Benchmarks & Model Evaluation](#machine-learning-benchmarks--model-evaluation)
  - [Regression Benchmark (Predicting Final Exam Score)](#1-regression-benchmark-predicting-final_exam_score)
  - [Classification Benchmark (Predicting Letter Grade)](#2-classification-benchmark-predicting-final_grade)
  - [Global Explainability & Feature Importance](#3-global-explainability--feature-importance)
- [Interactive Streamlit Web Dashboard](#interactive-streamlit-web-dashboard)
- [Repository File Structure](#repository-file-structure)
- [Step-by-Step Quickstart & Reproduction Guide](#step-by-step-quickstart--reproduction-guide)
- [Python API Reference & Programmatic Inference](#python-api-reference--programmatic-inference)
- [Generated Figures & Diagnostics Index](#generated-figures--diagnostics-index)
- [Future Roadmap](#future-roadmap)

---

## Project Overview & Key Objectives

Academic success is shaped by an intricate interplay of individual study routines (study time, sleep duration, attendance) and external environments (technology access, parental education level, employment commitments).

This project provides a comprehensive machine learning solution with the following core pillars:
1. **Exploratory Data Analysis (EDA):** Statistical profiling across 1,000 students to quantify academic drivers and equity gaps.
2. **Domain-Specific Feature Engineering:** Formulating composite metrics (study-to-sleep ratios, effort indices, academic momentum) to capture non-linear student behavior.
3. **Dual Supervised Machine Learning Engines:**
   - **Continuous Regression Engine:** Predicts exact score percentages ($0 - 100$) using regularized linear and tree-based models.
   - **Multi-Class Classification Engine:** Predicts categorical letter grades ($A, B, C, D, F$) with probability distributions.
4. **Early Warning & Prescription System:** Flags vulnerable students falling below engagement thresholds and recommends tailored recovery actions.
5. **Production Streamlit Dashboard:** An interactive web portal supporting real-time simulation, exploratory filtering, benchmark leaderboard inspection, and batch CSV scoring.

---

## Key Insights & Statistical Findings

Analysis of the 1,000 student records revealed critical statistical insights into performance drivers:

| Determinant | Metric / Correlation | Statistical Insight & Practical Impact |
| :--- | :--- | :--- |
| **Daily Study Time** | Pearson $r = +0.568$ | **Primary Driver:** The single strongest positive correlate of exam performance. Each additional daily study hour contributes approximately $+4.8$ to $+5.2$ points. |
| **Previous Academic Grade** | Pearson $r = +0.406$ | **Momentum Factor:** Prior mastery creates a strong performance floor; past performance correlates strongly with final outcomes. |
| **Lecture Attendance** | Pearson $r = +0.262$ | **Engagement Baseline:** Attendance below $75\%$ steeply increases the likelihood of receiving failing or near-failing grades ($D$ or $F$). |
| **Home Internet Access** | Mean Gap: $\Delta = +5.02\text{ pts}$ | Students with reliable internet access average **84.28**, compared to **79.26** for those without internet, underscoring the digital divide. |
| **Part-Time Employment** | Mean Gap: $\Delta = -3.29\text{ pts}$ | Working students average **81.29** vs **84.58** for non-working peers due to split time budgets and fatigue. |
| **Sleep Duration** | Optimal window | Peak performance occurs within **7.0 to 8.5 hours/night**. Extreme sleep restriction ($< 5\text{ hrs}$) degrades the return on study hours. |

---

## Dataset Schema & Architecture

The dataset includes **1,000 student records** across 12 initial attributes with zero missing values or duplicate rows.

### Data Dictionary

| Feature Name | Data Type | Category | Description & Valid Range |
| :--- | :--- | :--- | :--- |
| `student_id` | Identifier | Metadata | Unique identifier (`STUDENT_0001` to `STUDENT_1000`). |
| `gender` | Categorical | Demographic | Binary indicator: `Male`, `Female`. |
| `study_time_hours` | Numerical (Float) | Behavioral | Daily dedicated study hours ($0.5$ to $12.0\text{ hrs}$). |
| `attendance_percent` | Numerical (Float) | Behavioral | Percentage of classes attended ($35.0\%$ to $100.0\%$). |
| `sleep_hours` | Numerical (Float) | Behavioral | Average nightly sleep duration ($3.5$ to $10.5\text{ hrs}$). |
| `parental_education` | Categorical | Socio-demographic | Highest parental level: `None`, `High School`, `Bachelors`, `Masters`, `PhD`. |
| `internet_access` | Binary | Environmental | High-speed home connection: `Yes`, `No`. |
| `extracurricular_activities` | Binary | Behavioral | Participation in clubs/athletics: `Yes`, `No`. |
| `part_time_job` | Binary | Environmental | Student employment status: `Yes`, `No`. |
| `previous_grade` | Numerical (Float) | Historical | Prerequisite / midterm score ($25.0$ to $100.0$). |
| `final_exam_score` | Numerical (Float) | **Target (Regression)** | Continuous final exam grade ($0.0$ to $100.0$). |
| `final_grade` | Categorical | **Target (Classification)** | Standard letter grade: `A` ($\ge 90$), `B` ($80-89$), `C` ($70-79$), `D` ($60-69$), `F` ($< 60$). |

---

## Feature Engineering & Preprocessing Pipeline

The preprocessing pipeline (`src/preprocess.py`) engineers domain features and standardizes continuous/categorical features to optimize model generalization.

### 1. Mathematical Feature Formulations

$$\text{study\_to\_sleep\_ratio} = \frac{\text{study\_time\_hours}}{\text{sleep\_hours} + 10^{-5}}$$
*Measures study workload intensity relative to physiological recovery.*

$$\text{academic\_effort\_index} = \frac{\text{study\_time\_hours} \times \text{attendance\_percent}}{100.0}$$
*Captures composite behavioral investment across lecture attendance and independent study.*

$$\text{academic\_momentum} = \frac{\text{previous\_grade} \times \text{study\_time\_hours}}{10.0}$$
*Captures how prior mastery is amplified when combined with active study effort.*

$$\text{is\_at\_risk} = \begin{cases} 1 & \text{if } \text{attendance\_percent} < 75.0 \lor \text{study\_time\_hours} < 2.0 \\ 0 & \text{otherwise} \end{cases}$$
*Heuristic rule flagging students who fall below critical engagement thresholds.*

### 2. Scikit-Learn `ColumnTransformer` Architecture

- **Numerical Pipeline (7 features):** `study_time_hours`, `attendance_percent`, `sleep_hours`, `previous_grade`, `study_to_sleep_ratio`, `academic_effort_index`, `academic_momentum`
  - Imputation: `SimpleImputer(strategy='median')`
  - Scaling: `StandardScaler()`
- **Ordinal Categorical Pipeline (1 feature):** `parental_education`
  - Imputation: `SimpleImputer(fill_value='None')`
  - Encoding: `OrdinalEncoder(categories=[['None', 'High School', 'Bachelors', 'Masters', 'PhD']])`
- **Nominal Categorical Pipeline (4 features):** `gender`, `internet_access`, `extracurricular_activities`, `part_time_job`
  - Imputation: `SimpleImputer(strategy='most_frequent')`
  - Encoding: `OneHotEncoder(drop='first', sparse_output=False)`
- **Passthrough Feature:** `is_at_risk`

The fitted transformer is serialized to `models/preprocessor.joblib` for reproducible training and inference.

---

## Machine Learning Benchmarks & Model Evaluation

Evaluations were performed using an **80/20 train-test split** ($N_{\text{train}} = 800$, $N_{\text{test}} = 200$) with **5-Fold Cross-Validation** on the training split.

### 1. Regression Benchmark (Predicting `final_exam_score`)

| Rank | Model Architecture | 5-Fold CV $R^2$ (Mean $\pm$ Std) | Test MAE | Test RMSE | Test $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Lasso Regression (Tuned $\alpha=0.1$)** | $\mathbf{0.6552 \pm 0.0410}$ | $\mathbf{4.9333}$ | $\mathbf{6.3673}$ | $\mathbf{0.6233}$ |
| 🥈 | **Ridge Regression ($\alpha=1.0$)** | $0.6556 \pm 0.0387$ | $5.0158$ | $6.4614$ | $0.6121$ |
| 🥉 | **Linear Regression (OLS)** | $0.6551 \pm 0.0384$ | $5.0301$ | $6.4725$ | $0.6108$ |
| 4 | **Random Forest Regressor** | $0.5758 \pm 0.0548$ | $5.1690$ | $6.7036$ | $0.5825$ |
| 5 | **Support Vector Regressor (SVR)** | $0.6018 \pm 0.0421$ | $5.1662$ | $6.7331$ | $0.5788$ |
| 6 | **Gradient Boosting Regressor** | $0.5873 \pm 0.0558$ | $5.2384$ | $6.7554$ | $0.5760$ |
| 7 | **Hist Gradient Boosting** | $0.5944 \pm 0.0448$ | $5.3095$ | $6.8020$ | $0.5702$ |
| 8 | **Decision Tree Regressor** | $0.3754 \pm 0.0502$ | $5.9191$ | $7.4996$ | $0.4775$ |

> **Production Regressor:** **Lasso Regression** achieved the lowest Mean Absolute Error ($4.933$ pts) and highest test $R^2$ ($0.6233$), providing robust generalization without overfitting.

---

### 2. Classification Benchmark (Predicting `final_grade`)

| Rank | Classifier Model | 5-Fold CV Accuracy | Test Accuracy | Macro F1 | Weighted F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting Classifier** | $0.4938$ | $\mathbf{56.00\%}$ | $0.4032$ | $\mathbf{0.5545}$ |
| 🥈 | **Random Forest Classifier** | $0.4825$ | $55.50\%$ | $\mathbf{0.4161}$ | $0.5562$ |
| 🥉 | **Logistic Regression (Multinomial)** | $\mathbf{0.5175}$ | $55.00\%$ | $0.4277$ | $0.5623$ |
| 4 | **Decision Tree Classifier** | $0.4362$ | $39.50\%$ | $0.3305$ | $0.4154$ |

> **Production Classifier:** **Gradient Boosting Classifier** delivered the highest test accuracy ($56.00\%$) and provides well-calibrated class probability estimates across letter grades $A$ through $F$.

---

### 3. Global Explainability & Feature Importance

Extracting standardized coefficients and tree feature importances reveals the dominant drivers:

```
study_time_hours        ████████████████████████████  9.06
internet_access_Yes     ██████████████████            5.96
previous_grade          ███████████████               5.03
attendance_percent      ████████████                  3.94
part_time_job_Yes       ███████████                   3.52
academic_effort_index   █████████                     2.91
sleep_hours             █████                         1.79
```

---

## Interactive Streamlit Web Dashboard

The web application (`app/app.py`) provides an interactive interface organized across four functional modules:

```
┌────────────────────────────────────────────────────────────────────────┐
│               Student Academic Performance Analytics Dashboard         │
├───────────────────┬───────────────────┬────────────────┬───────────────┤
│ 1. Student        │ 2. Cohort EDA     │ 3. Model       │ 4. Batch CSV  │
│    Simulator      │    Explorer       │    Leaderboard │    Scoring    │
└───────────────────┴───────────────────┴────────────────┴───────────────┘
```

### Dashboard Tabs Overview:

1. **Student Predictor & Risk Advisory (Tab 1)**
   - **Interactive Sliders & Dropdowns:** Adjust study hours, sleep, attendance, previous grades, parental education, and job status.
   - **Real-Time Predictions:** Instant calculation of predicted final score, letter grade badge ($A, B, C, D, F$), academic risk status (`ON TRACK` vs `AT RISK`), and cohort benchmark delta.
   - **Probability Breakdown:** Visual bar chart showing the predicted probability for every letter grade.
   - **Tailored Interventions:** Dynamic academic recommendations based on student-specific weak points.

2. **Cohort Exploratory Data Analysis (Tab 2)**
   - **Interactive Cohort Slicing:** Multi-select filtering by final grade, employment status, and internet access.
   - **Dynamic Visuals:** Score distribution histograms, study-time regression trendlines, and categorical heatmaps.

3. **Model Leaderboard & Explainability (Tab 3)**
   - **Benchmark Comparison Tables:** Detailed metrics for both regression and classification models.
   - **Global Feature Importance Chart:** Visual ranking of top decision drivers.
   - **Diagnostic Figures:** Embedded residual plots and confusion matrices.

4. **Batch CSV Scoring & Export (Tab 4)**
   - **Bulk Processing:** Upload any student roster in CSV format or test sample records.
   - **Instant Inference:** Scores all students simultaneously and adds predicted score, predicted grade, risk status, and actionable recommendations.
   - **One-Click Download:** Export the enriched roster as a CSV file.

---

## Repository File Structure

```text
Student Performance Project/
│
├── data/
│   ├── student_performance_dataset.csv     # Raw dataset (1,000 student records)
│   └── processed/                          # Scaled, encoded train/test datasets
│       ├── train_regression.csv            # 800 training rows (13 transformed features)
│       ├── test_regression.csv             # 200 holdout testing rows
│       ├── train_classification.csv        # Categorical classification train split
│       └── test_classification.csv         # Categorical classification test split
│
├── notebooks/                              # Interactive research notebooks
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis & visual discovery
│   ├── 02_preprocessing.ipynb              # Pipeline design & transformation experiments
│   └── 03_modeling.ipynb                   # Model training, hyperparameter tuning & evaluation
│
├── src/                                    # Modular production source code
│   ├── __init__.py
│   ├── data_loader.py                      # Robust data ingestion & path resolution
│   ├── eda_analysis.py                     # Automated statistical analysis & figure generation
│   ├── preprocess.py                       # Data cleaning, encoding, scaling & pipeline export
│   ├── train.py                            # Model training, cross-validation & artifact persistence
│   └── predict.py                          # Real-time inference & recommendation engine
│
├── models/                                 # Serialized production artifacts
│   ├── preprocessor.joblib                 # Fitted Scikit-Learn ColumnTransformer pipeline
│   ├── best_regression_model.joblib        # Fitted Lasso Regressor model
│   └── best_classification_model.joblib    # Fitted Gradient Boosting Classifier model
│
├── reports/                                # Analytics & benchmark outputs
│   ├── model_evaluation_metrics.json       # Consolidated benchmark evaluation data
│   └── figures/                            # Publication-grade analytical figures (01-09)
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
│   └── app.py                              # Full-featured Streamlit web application
│
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

---

## Step-by-Step Quickstart & Reproduction Guide

### 1. Set Up Environment

```bash
# Clone the repository
git clone https://github.com/KochenkExe/Student-Performance-Project.git
cd "Student Performance Project"

# Create a virtual environment (optional)
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Step 1: Exploratory Data Analysis (EDA)
Generates statistical summaries and saves figures to `reports/figures/`:
```bash
python src/eda_analysis.py
```

### 3. Step 2: Feature Engineering & Preprocessing
Cleans raw data, engineers domain features, creates splits, and saves `models/preprocessor.joblib`:
```bash
python src/preprocess.py
```

### 4. Step 3: Model Training & Evaluation
Trains regressors and classifiers, evaluates benchmarks, saves top models, and exports `reports/model_evaluation_metrics.json`:
```bash
python src/train.py
```

### 5. Step 4: Launch the Streamlit Web Dashboard
Launches the interactive dashboard in your default browser:
```bash
python -m streamlit run app/app.py
```
*(Or `streamlit run app/app.py` if Streamlit is registered in your system PATH)*

---

## Python API Reference & Programmatic Inference

You can import the prediction engine directly into external Python applications:

```python
from src.predict import StudentPerformancePredictor

# Initialize predictor (loads saved models and preprocessor)
predictor = StudentPerformancePredictor()

# Define student payload
student = {
    "gender": "Female",
    "study_time_hours": 4.5,
    "attendance_percent": 92.0,
    "sleep_hours": 7.5,
    "parental_education": "Bachelors",
    "internet_access": "Yes",
    "extracurricular_activities": "Yes",
    "part_time_job": "No",
    "previous_grade": 80.0
}

# Run prediction
result = predictor.predict_single(student)

print(f"Predicted Final Score: {result['predicted_final_score']} / 100")
print(f"Predicted Grade      : {result['predicted_letter_grade']}")
print(f"Academic Risk Status : {'AT RISK' if result['is_at_risk'] else 'ON TRACK'}")
print(f"Grade Probabilities  : {result['grade_probabilities']}")
print(f"Recommendations      : {result['actionable_recommendations']}")
```

---

## Generated Figures & Diagnostics Index

All visual figures are stored in `reports/figures/`:

| Figure Name | Description |
| :--- | :--- |
| `01_target_distributions.png` | Distribution histograms for final exam scores and letter grade counts ($A-F$). |
| `02_numeric_distributions.png` | Univariate distribution histograms with KDE for continuous input features. |
| `03_correlation_heatmap.png` | Pearson correlation matrix highlighting linear relationships with target variables. |
| `04_key_relationships_regression.png` | Scatter plots with regression lines for study time, attendance, and prior grades. |
| `05_categorical_impact_boxplots.png` | Boxplots displaying score spreads segmented by internet access, job status, and education. |
| `06_model_comparison_regression.png` | Bar chart comparing $R^2$, MAE, and RMSE across all 8 evaluated regression models. |
| `07_actual_vs_predicted_residuals.png` | Parity plot (Actual vs Predicted) and residual scatter plot for the best regressor. |
| `08_feature_importance.png` | Ranked horizontal bar chart of global feature importance weights. |
| `09_confusion_matrix_classification.png` | Confusion matrix heatmap comparing true vs predicted letter grades. |

---

## Future Roadmap

- [ ] **Localized SHAP Force / Waterfall Plots:** Add personalized feature contribution waterfalls in the Streamlit Single Student simulator.
- [ ] **FastAPI Backend Integration:** Expose `/predict` and `/batch-predict` endpoints as a standalone REST API.
- [ ] **Docker Containerization:** Add Dockerfile and docker-compose for one-command deployment to cloud providers.
- [ ] **Longitudinal Multi-Semester Tracking:** Enable time-series tracking of student grade trajectories over multiple semesters.

---

## License

This project is licensed under the [MIT License](LICENSE).

