import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="data/student_performance_dataset.csv", output_dir="reports/figures"):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the student performance dataset.
    Saves visual plots to output_dir and prints summary insights.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("=" * 60)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"\n[+] Dataset Loaded Successfully!")
    print(f"    - Total Rows: {df.shape[0]}")
    print(f"    - Total Columns: {df.shape[1]}")
    
    # 2. Data Overview & Hygiene Check
    print("\n--- Data Structure & Missing Values ---")
    print(df.info())
    
    missing_vals = df.isnull().sum()
    print(f"\nMissing values per column:\n{missing_vals[missing_vals > 0] if missing_vals.sum() > 0 else 'No missing values found!'}")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # 3. Summary Statistics
    num_cols = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade', 'final_exam_score']
    cat_cols = ['gender', 'parental_education', 'internet_access', 'extracurricular_activities', 'part_time_job', 'final_grade']
    
    print("\n--- Numerical Summary Statistics ---")
    print(df[num_cols].describe().round(2))
    
    print("\n--- Categorical Distributions ---")
    for col in cat_cols:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + '% (' + df[col].value_counts().astype(str) + ')')
        
    # Styling configuration
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("viridis")
    
    # 4. Plot 1: Target Variable Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['final_exam_score'], kde=True, color='#2b5c8f', bins=20, ax=axes[0])
    axes[0].set_title("Distribution of Final Exam Score", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Final Exam Score (0 - 100)")
    axes[0].set_ylabel("Student Count")
    
    grade_order = ['A', 'B', 'C', 'D', 'F']
    present_grades = [g for g in grade_order if g in df['final_grade'].values]
    sns.countplot(data=df, x='final_grade', order=present_grades, palette="Set2", ax=axes[1])
    axes[1].set_title("Distribution of Letter Grades", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Final Grade")
    axes[1].set_ylabel("Student Count")
    
    # Add count labels on bars
    for p in axes[1].patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            axes[1].annotate(f'{int(height)}',
                             (p.get_x() + p.get_width() / 2., height / 2),
                             ha='center', va='center', fontsize=11, color='white', fontweight='bold')
            
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "01_target_distributions.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot1_path}")
    
    # 5. Plot 2: Numerical Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    features_to_plot = ['study_time_hours', 'attendance_percent', 'sleep_hours', 'previous_grade']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    
    for i, col in enumerate(features_to_plot):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=20)
        axes[i].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontweight='bold')
        axes[i].set_xlabel(col.replace('_', ' ').title())
        axes[i].set_ylabel("Count")
        
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "02_numeric_distributions.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot2_path}")
    
    # 6. Plot 3: Correlation Matrix
    plt.figure(figsize=(9, 7))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, square=True, linewidths=1)
    plt.title("Correlation Matrix (Pearson)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "03_correlation_heatmap.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot3_path}")
    
    # 7. Plot 4: Key Feature Relationships with Final Exam Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.regplot(data=df, x='study_time_hours', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#2980b9'}, line_kws={'color': '#c0392b', 'linewidth': 2}, ax=axes[0])
    axes[0].set_title("Study Time vs Final Exam Score", fontweight='bold')
    axes[0].set_xlabel("Study Time (Hours / Day)")
    axes[0].set_ylabel("Final Exam Score")
    
    sns.regplot(data=df, x='attendance_percent', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#27ae60'}, line_kws={'color': '#c0392b', 'linewidth': 2}, ax=axes[1])
    axes[1].set_title("Attendance % vs Final Exam Score", fontweight='bold')
    axes[1].set_xlabel("Attendance (%)")
    axes[1].set_ylabel("Final Exam Score")
    
    sns.regplot(data=df, x='previous_grade', y='final_exam_score', scatter_kws={'alpha': 0.5, 'color': '#8e44ad'}, line_kws={'color': '#c0392b', 'linewidth': 2}, ax=axes[2])
    axes[2].set_title("Previous Grade vs Final Exam Score", fontweight='bold')
    axes[2].set_xlabel("Previous Grade")
    axes[2].set_ylabel("Final Exam Score")
    
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "04_key_relationships_regression.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot4_path}")
    
    # 8. Plot 5: Categorical Factor Impact on Final Exam Score
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cat_factors = [
        ('parental_education', ['None', 'High School', 'Bachelors', 'Masters', 'PhD']),
        ('part_time_job', ['No', 'Yes']),
        ('internet_access', ['No', 'Yes']),
        ('extracurricular_activities', ['No', 'Yes']),
        ('gender', ['Male', 'Female'])
    ]
    
    for idx, (factor, order) in enumerate(cat_factors):
        row, col = divmod(idx, 3)
        sns.boxplot(data=df, x=factor, y='final_exam_score', order=order, ax=axes[row, col], palette="Set3")
        sns.stripplot(data=df, x=factor, y='final_exam_score', order=order, ax=axes[row, col], color='black', alpha=0.15, jitter=0.2)
        axes[row, col].set_title(f"Score by {factor.replace('_', ' ').title()}", fontweight='bold')
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("Final Exam Score")
        
    # Hide the unused 6th subplot
    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    plot5_path = os.path.join(output_dir, "05_categorical_impact_boxplots.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot5_path}")
    
    # 9. Print Key Analytical Insights
    print("\n" + "=" * 60)
    print("KEY EDA INSIGHTS & STATISTICAL HIGHLIGHTS")
    print("=" * 60)
    
    print("\n1. Correlations with Final Exam Score:")
    for feat in ['study_time_hours', 'attendance_percent', 'previous_grade', 'sleep_hours']:
        r = df[feat].corr(df['final_exam_score'])
        print(f"   - {feat:20s}: r = {r:.4f}")
        
    print("\n2. Mean Score by Parental Education:")
    parent_mean = df.groupby('parental_education')['final_exam_score'].agg(['mean', 'std', 'count']).round(2)
    print(parent_mean)
    
    print("\n3. Mean Score by Part-time Job:")
    job_mean = df.groupby('part_time_job')['final_exam_score'].agg(['mean', 'std', 'count']).round(2)
    print(job_mean)
    
    print("\n4. Mean Score by Internet Access:")
    net_mean = df.groupby('internet_access')['final_exam_score'].agg(['mean', 'std', 'count']).round(2)
    print(net_mean)
    
    print("\n5. Mean Score by Gender:")
    gender_mean = df.groupby('gender')['final_exam_score'].agg(['mean', 'std', 'count']).round(2)
    print(gender_mean)
    
    print("\n[+] EDA Execution Complete. All figures saved to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    run_eda()
