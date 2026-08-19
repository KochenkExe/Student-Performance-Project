import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold

def train_and_evaluate_regression(
    train_path="data/processed/train_regression.csv", 
    test_path="data/processed/test_regression.csv",
    output_dir="reports/figures",
    models_dir="models"
):
    """
    Trains, evaluates, tunes, and saves regression models for predicting final_exam_score.
    """
    print("\n" + "=" * 60)
    print("STEP 3A: REGRESSION MODELING (Target: final_exam_score)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['final_exam_score']).values
    y_train = train_df['final_exam_score'].values
    X_test = test_df.drop(columns=['final_exam_score']).values
    y_test = test_df['final_exam_score'].values
    feature_names = [c for c in train_df.columns if c != 'final_exam_score']

    print(f"[+] Loaded Processed Data: Train={X_train.shape}, Test={X_test.shape}")

    # 2. Define Model Portfolio
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=4, random_state=42),
        "Support Vector Regressor": SVR(C=10.0, epsilon=0.2)
    }

    results = []
    trained_models = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Cross-validation on Train Set
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        
        # Fit on Full Training Set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Test set evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "CV R2 (Mean)": round(cv_scores.mean(), 4),
            "CV R2 (Std)": round(cv_scores.std(), 4),
            "Test MAE": round(mae, 4),
            "Test RMSE": round(rmse, 4),
            "Test R2": round(r2, 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    print("\n--- Regression Benchmark Results ---")
    print(results_df.to_string(index=False))

    # 3. Hyperparameter Tuning on Best Performing Model
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n[+] Best Baseline Model: {best_model_name}")
    print("[*] Initiating Hyperparameter Optimization with GridSearchCV...")

    if "Gradient Boosting" in best_model_name or "Hist" in best_model_name:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=kf, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_tuned_model = grid.best_estimator_
        print(f"    - Best Params: {grid.best_params_}")
    elif "Random Forest" in best_model_name:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=kf, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_tuned_model = grid.best_estimator_
        print(f"    - Best Params: {grid.best_params_}")
    else:
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
        grid = GridSearchCV(Ridge(random_state=42), param_grid, cv=kf, scoring='r2')
        grid.fit(X_train, y_train)
        best_tuned_model = grid.best_estimator_
        print(f"    - Best Params: {grid.best_params_}")

    # Evaluate Tuned Model
    y_test_pred = best_tuned_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, y_test_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    tuned_r2 = r2_score(y_test, y_test_pred)

    print("\n--- Final Tuned Model Performance on Test Set ---")
    print(f"    - Test MAE : {tuned_mae:.4f}")
    print(f"    - Test RMSE: {tuned_rmse:.4f}")
    print(f"    - Test R^2 : {tuned_r2:.4f} ({tuned_r2*100:.2f}% variance explained)")

    # Save Best Regression Model
    best_reg_path = os.path.join(models_dir, "best_regression_model.joblib")
    joblib.dump(best_tuned_model, best_reg_path)
    print(f"[+] Saved Best Regression Model to: {best_reg_path}")

    # 4. Generate Visual Plots
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(data=results_df, x='Test R2', y='Model', palette='Blues_r', ax=ax)
    ax.set_title("Regression Model Comparison (Test R² Score)", fontsize=14, fontweight='bold')
    ax.set_xlabel("R² Score (Higher is Better)")
    ax.set_xlim(0, 1.0)
    for p in bars.patches:
        width = p.get_width()
        ax.annotate(f'{width:.4f}', (width + 0.01, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "06_model_comparison_regression.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot1_path}")

    # Plot 2: Actual vs Predicted and Residuals Plot
    residuals = y_test - y_test_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter: Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='#2980b9', edgecolors='k', s=50)
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal 1:1 Line")
    axes[0].set_title(f"Actual vs Predicted Exam Scores (R² = {tuned_r2:.3f})", fontweight='bold')
    axes[0].set_xlabel("Actual Exam Score")
    axes[0].set_ylabel("Predicted Exam Score")
    axes[0].legend()

    # Residuals distribution
    sns.histplot(residuals, kde=True, color='#e74c3c', bins=20, ax=axes[1])
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title("Residuals Distribution (Errors)", fontweight='bold')
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "07_actual_vs_predicted_residuals.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot2_path}")

    # Plot 3: Feature Importance
    if hasattr(best_tuned_model, 'feature_importances_'):
        importances = best_tuned_model.feature_importances_
    elif hasattr(best_tuned_model, 'coef_'):
        importances = np.abs(best_tuned_model.coef_)
    else:
        importances = np.ones(len(feature_names))

    feat_imp_df = pd.DataFrame({
        'Feature': [f.replace('num__', '').replace('cat__', '').replace('ord__', '').replace('pass__', '') for f in feature_names],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance in Final Prediction Model", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "08_feature_importance.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot3_path}")

    return results_df, best_tuned_model, feat_imp_df


def train_and_evaluate_classification(
    train_path="data/processed/train_classification.csv", 
    test_path="data/processed/test_classification.csv",
    output_dir="reports/figures",
    models_dir="models"
):
    """
    Trains, evaluates, and saves classification models for predicting final_grade (A, B, C, D, F).
    """
    print("\n" + "=" * 60)
    print("STEP 3B: CLASSIFICATION MODELING (Target: final_grade)")
    print("=" * 60)

    # 1. Load Data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['final_grade']).values
    y_train = train_df['final_grade'].values.astype(int)
    X_test = test_df.drop(columns=['final_grade']).values
    y_test = test_df['final_grade'].values.astype(int)

    grade_labels = ['F', 'D', 'C', 'B', 'A']
    present_classes = np.unique(np.concatenate([y_train, y_test]))
    class_names = [grade_labels[i] for i in present_classes]

    # 2. Train Classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }

    clf_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        cv_acc = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        clf_results.append({
            "Model": name,
            "CV Accuracy": round(cv_acc.mean(), 4),
            "Test Accuracy": round(acc, 4),
            "F1 (Macro)": round(f1_macro, 4),
            "F1 (Weighted)": round(f1_weighted, 4)
        })

    clf_results_df = pd.DataFrame(clf_results).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
    print("\n--- Classification Benchmark Results ---")
    print(clf_results_df.to_string(index=False))

    # Pick Best Classifier
    best_clf_name = clf_results_df.iloc[0]["Model"]
    best_clf = classifiers[best_clf_name]
    best_clf_path = os.path.join(models_dir, "best_classification_model.joblib")
    joblib.dump(best_clf, best_clf_path)
    print(f"\n[+] Saved Best Classification Model ({best_clf_name}) to: {best_clf_path}")

    # Plot Confusion Matrix
    y_test_pred = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=present_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({best_clf_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "09_confusion_matrix_classification.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {cm_path}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))

    return clf_results_df, best_clf


def run_full_training_pipeline():
    """
    Runs both regression and classification pipelines and saves metrics summary.
    """
    reg_results, best_reg, feat_imp = train_and_evaluate_regression()
    clf_results, best_clf = train_and_evaluate_classification()

    # Save metrics summary to JSON
    summary = {
        "regression_results": reg_results.to_dict(orient="records"),
        "classification_results": clf_results.to_dict(orient="records"),
        "top_features": feat_imp.head(7).to_dict(orient="records")
    }

    with open("reports/model_evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[+] Model evaluation metrics exported to: reports/model_evaluation_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    run_full_training_pipeline()
