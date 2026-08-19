import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 2: Data Preprocessing & Feature Engineering\n",
    "## Student Performance Analysis & Prediction\n",
    "\n",
    "### Objectives:\n",
    "1. Impute missing values and clean data.\n",
    "2. Engineer domain-specific interaction features (`study_to_sleep_ratio`, `academic_effort_index`, `academic_momentum`, `is_at_risk`).\n",
    "3. Encode categorical features (Ordinal for education, One-Hot for binary demographics).\n",
    "4. Standardize numerical features using `StandardScaler`.\n",
    "5. Split data into 80/20 train/test sets without data leakage.\n",
    "6. Serialize the fitted preprocessing pipeline for downstream inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "sys.path.append(\"..\")\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from src.preprocess import DataPreprocessor, run_preprocessing_pipeline\n",
    "\n",
    "print(\"Modules imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Inspect Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = \"../data/student_performance_dataset.csv\" if os.path.exists(\"../data/student_performance_dataset.csv\") else \"student_performance_dataset.csv\"\n",
    "df_raw = pd.read_csv(data_path)\n",
    "\n",
    "preprocessor = DataPreprocessor(target_type=\"regression\")\n",
    "df_engineered = preprocessor.clean_and_engineer_features(df_raw)\n",
    "\n",
    "print(\"Engineered Features Sample:\")\n",
    "df_engineered[['study_time_hours', 'sleep_hours', 'study_to_sleep_ratio', 'academic_effort_index', 'academic_momentum', 'is_at_risk']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Execute Transformation Pipeline & Train/Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df_raw, test_size=0.2, random_state=42)\n",
    "\n",
    "print(f\"Feature count: {len(feature_names)}\")\n",
    "print(f\"Training set: X={X_train.shape}, y={y_train.shape}\")\n",
    "print(f\"Testing set : X={X_test.shape}, y={y_test.shape}\")\n",
    "\n",
    "# Display sample transformed dataframe\n",
    "df_train_transformed = pd.DataFrame(X_train, columns=feature_names)\n",
    "df_train_transformed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Save Processed Artifacts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessor.save_pipeline(\"../models/preprocessor.joblib\")\n",
    "print(\"Pipeline successfully exported for deployment!\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("notebooks/02_preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("notebooks/02_preprocessing.ipynb generated successfully!")
