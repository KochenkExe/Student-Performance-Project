import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 1: Exploratory Data Analysis (EDA)\n",
    "## Student Performance Analysis & Prediction\n",
    "\n",
    "### Project Objective\n",
    "Understand the demographic, behavioral, and academic drivers influencing student success (`final_exam_score` and `final_grade`).\n",
    "\n",
    "---\n",
    "### Table of Contents\n",
    "1. **Environment Setup & Data Ingestion**\n",
    "2. **Data Structure & Missing Value Analysis**\n",
    "3. **Statistical Summary (Numerical & Categorical)**\n",
    "4. **Target Variable Distributions**\n",
    "5. **Correlation & Multivariable Analysis**\n",
    "6. **Categorical Factors & Demographic Disparities**\n",
    "7. **Key Findings & Next Steps (Feature Engineering & Modeling)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style\n",
    "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
    "print(\"Libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "data_path = \"../data/student_performance_dataset.csv\" if os.path.exists(\"../data/student_performance_dataset.csv\") else \"student_performance_dataset.csv\"\n",
    "df = pd.read_csv(data_path)\n",
    "\n",
    "print(f\"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Structure & Data Hygiene Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data types and non-null counts\n",
    "df.info()\n",
    "\n",
    "# Missing values and duplicates check\n",
    "missing = df.isnull().sum()\n",
    "print(\"\\n--- Missing Values ---\")\n",
    "print(missing[missing > 0] if missing.sum() > 0 else \"No missing values!\")\n",
    "print(f\"\\nDuplicate rows: {df.duplicated().sum()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade', 'final_exam_score']\n",
    "df[num_cols].describe().T.round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Target Variable Analysis: Exam Scores & Grades"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Final Exam Score Distribution\n",
    "sns.histplot(df['final_exam_score'], kde=True, color='#2b5c8f', bins=20, ax=axes[0])\n",
    "axes[0].set_title(\"Distribution of Final Exam Score\", fontweight='bold')\n",
    "axes[0].set_xlabel(\"Final Exam Score (0 - 100)\")\n",
    "\n",
    "# Final Grade Distribution\n",
    "grade_order = ['A', 'B', 'C', 'D', 'F']\n",
    "present_grades = [g for g in grade_order if g in df['final_grade'].values]\n",
    "sns.countplot(data=df, x='final_grade', order=present_grades, palette=\"Set2\", hue='final_grade', legend=False, ax=axes[1])\n",
    "axes[1].set_title(\"Distribution of Letter Grades\", fontweight='bold')\n",
    "axes[1].set_xlabel(\"Final Grade\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 6))\n",
    "corr = df[num_cols].corr()\n",
    "sns.heatmap(corr, annot=True, fmt=\".3f\", cmap=\"coolwarm\", square=True, linewidths=1)\n",
    "plt.title(\"Correlation Matrix (Pearson)\", fontweight='bold', pad=12)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Relationships Between Predictors and Final Exam Score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "# Study Time vs Score\n",
    "sns.regplot(data=df, x='study_time_hours', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#2980b9'}, line_kws={'color': '#c0392b'}, ax=axes[0])\n",
    "axes[0].set_title(\"Study Time vs Final Exam Score (r = 0.568)\", fontweight='bold')\n",
    "axes[0].set_xlabel(\"Study Time (Hours / Day)\")\n",
    "\n",
    "# Attendance vs Score\n",
    "sns.regplot(data=df, x='attendance_percent', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#27ae60'}, line_kws={'color': '#c0392b'}, ax=axes[1])\n",
    "axes[1].set_title(\"Attendance % vs Final Exam Score (r = 0.262)\", fontweight='bold')\n",
    "axes[1].set_xlabel(\"Attendance (%)\")\n",
    "\n",
    "# Previous Grade vs Score\n",
    "sns.regplot(data=df, x='previous_grade', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#8e44ad'}, line_kws={'color': '#c0392b'}, ax=axes[2])\n",
    "axes[2].set_title(\"Previous Grade vs Final Exam Score (r = 0.406)\", fontweight='bold')\n",
    "axes[2].set_xlabel(\"Previous Grade\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Categorical & Demographic Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "\n",
    "# Internet Access\n",
    "sns.boxplot(data=df, x='internet_access', y='final_exam_score', palette=\"Pastel1\", hue='internet_access', legend=False, ax=axes[0, 0])\n",
    "axes[0, 0].set_title(\"Impact of Internet Access on Final Score\", fontweight='bold')\n",
    "\n",
    "# Part-time Job\n",
    "sns.boxplot(data=df, x='part_time_job', y='final_exam_score', palette=\"Pastel2\", hue='part_time_job', legend=False, ax=axes[0, 1])\n",
    "axes[0, 1].set_title(\"Impact of Part-time Job on Final Score\", fontweight='bold')\n",
    "\n",
    "# Parental Education\n",
    "edu_order = ['High School', 'Bachelors', 'Masters', 'PhD']\n",
    "sns.boxplot(data=df, x='parental_education', y='final_exam_score', order=edu_order, palette=\"Set3\", hue='parental_education', legend=False, ax=axes[1, 0])\n",
    "axes[1, 0].set_title(\"Impact of Parental Education\", fontweight='bold')\n",
    "\n",
    "# Extracurricular Activities\n",
    "sns.boxplot(data=df, x='extracurricular_activities', y='final_exam_score', palette=\"Accent\", hue='extracurricular_activities', legend=False, ax=axes[1, 1])\n",
    "axes[1, 1].set_title(\"Impact of Extracurricular Activities\", fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
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

with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("notebooks/01_eda.ipynb generated successfully!")
