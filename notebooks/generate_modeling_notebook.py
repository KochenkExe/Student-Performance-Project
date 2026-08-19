import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 3: Model Training, Evaluation & Tuning\n",
    "## Student Performance Analysis & Prediction\n",
    "\n",
    "### Objectives:\n",
    "1. **Regression Benchmark**: Compare Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting, and SVR.\n",
    "2. **Hyperparameter Tuning**: Optimize the top-performing regressor with K-Fold Cross-Validation.\n",
    "3. **Model Explainability**: Analyze feature importance and residual distributions.\n",
    "4. **Classification Benchmark**: Train multi-class classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) for letter grade prediction.\n",
    "5. **Serialization**: Save the best performing models for inference and deployment."
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
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from src.train import train_and_evaluate_regression, train_and_evaluate_classification\n",
    "\n",
    "print(\"Training modules loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Regression: Predict Final Exam Score (0 - 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "reg_results, best_reg_model, feat_imp = train_and_evaluate_regression(\n",
    "    train_path=\"../data/processed/train_regression.csv\",\n",
    "    test_path=\"../data/processed/test_regression.csv\",\n",
    "    output_dir=\"../reports/figures\",\n",
    "    models_dir=\"../models\"\n",
    ")\n",
    "\n",
    "reg_results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Importance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Top Most Important Predictive Features:\")\n",
    "feat_imp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Classification: Predict Final Letter Grade (A, B, C, D, F)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_results, best_clf_model = train_and_evaluate_classification(\n",
    "    train_path=\"../data/processed/train_classification.csv\",\n",
    "    test_path=\"../data/processed/test_classification.csv\",\n",
    "    output_dir=\"../reports/figures\",\n",
    "    models_dir=\"../models\"\n",
    ")\n",
    "\n",
    "clf_results"
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

with open("notebooks/03_modeling.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("notebooks/03_modeling.ipynb generated successfully!")
