import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add project root to sys.path so src imports work reliably
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.preprocess import DataPreprocessor
from src.predict import StudentPerformancePredictor

# Set Streamlit page config
st.set_page_config(
    page_title="Student Performance AI & Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .grade-badge-a { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 8px; font-weight: bold; }
    .grade-badge-b { background-color: #3B82F6; color: white; padding: 4px 12px; border-radius: 8px; font-weight: bold; }
    .grade-badge-c { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 8px; font-weight: bold; }
    .grade-badge-d { background-color: #F97316; color: white; padding: 4px 12px; border-radius: 8px; font-weight: bold; }
    .grade-badge-f { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return StudentPerformancePredictor()

@st.cache_data
def get_dataset():
    return load_data()

@st.cache_data
def get_metrics_summary():
    metrics_path = "reports/model_evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

def main():
    # Load resources
    df_raw = get_dataset()
    predictor = get_predictor()
    metrics_data = get_metrics_summary()

    # Header
    st.markdown('<div class="main-header">Student Academic Performance Analytics & AI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict final exam scores, analyze risk factors, and explore demographic and behavioral drivers.</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Dataset Summary")
        st.write(f"**Total Students:** {len(df_raw):,}")
        st.write(f"**Average Exam Score:** {df_raw['final_exam_score'].mean():.2f} / 100")
        st.write(f"**Average Attendance:** {df_raw['attendance_percent'].mean():.1f}%")
        st.write(f"**Avg Daily Study:** {df_raw['study_time_hours'].mean():.1f} hrs")
        st.markdown("---")
        st.caption("Developed with Scikit-Learn, Pandas and Streamlit")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Student Predictor & Risk Advisory",
        "Interactive EDA Dashboard",
        "Model Leaderboard & Explainability",
        "Batch CSV Scoring"
    ])

    # -------------------------------------------------------------
    # TAB 1: INDIVIDUAL PREDICTION
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Simulate Student Profile & Predict Outcome")
        st.markdown("Adjust the academic and lifestyle parameters below to generate an AI-powered score estimate and personalized recommendations.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Academic Habits")
            study_time = st.slider("Daily Study Time (Hours)", min_value=0.0, max_value=12.0, value=4.0, step=0.1)
            attendance = st.slider("Attendance Rate (%)", min_value=30.0, max_value=100.0, value=88.0, step=0.5)
            prev_grade = st.slider("Previous Academic Grade", min_value=20.0, max_value=100.0, value=75.0, step=0.5)

        with col2:
            st.markdown("#### Lifestyle & Environment")
            sleep_hours = st.slider("Sleep Duration (Hours/Night)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
            internet = st.selectbox("Internet Access at Home?", ["Yes", "No"], index=0)
            part_time = st.selectbox("Has Part-Time Employment?", ["No", "Yes"], index=0)

        with col3:
            st.markdown("#### Background & Extracurriculars")
            parent_edu = st.selectbox("Parental Education Level", ["High School", "Bachelors", "Masters", "PhD", "None"], index=1)
            extracurricular = st.selectbox("Participates in Extracurriculars?", ["Yes", "No"], index=0)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)

        student_payload = {
            "gender": gender,
            "study_time_hours": study_time,
            "attendance_percent": attendance,
            "sleep_hours": sleep_hours,
            "parental_education": parent_edu,
            "internet_access": internet,
            "extracurricular_activities": extracurricular,
            "part_time_job": part_time,
            "previous_grade": prev_grade
        }

        if st.button("Run Prediction & Analysis", type="primary", use_container_width=True):
            result = predictor.predict_single(student_payload)

            st.markdown("---")
            st.subheader("Prediction Output")

            # Metrics Row
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Exam Score</div>
                    <div class="metric-val">{result['predicted_final_score']}</div>
                    <small>Out of 100.0</small>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                grade = result['predicted_letter_grade']
                badge_class = f"grade-badge-{grade.lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Letter Grade</div>
                    <div class="metric-val"><span class="{badge_class}">{grade}</span></div>
                    <small>Classification Output</small>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                status_text = "AT RISK" if result['is_at_risk'] else "ON TRACK"
                status_color = "#DC2626" if result['is_at_risk'] else "#16A34A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Academic Risk Status</div>
                    <div class="metric-val" style="color: {status_color}; font-size: 1.5rem;">{status_text}</div>
                    <small>Attendance / Study Filter</small>
                </div>
                """, unsafe_allow_html=True)

            with res_col4:
                diff = result['predicted_final_score'] - df_raw['final_exam_score'].mean()
                diff_sign = "+" if diff >= 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Benchmark Delta</div>
                    <div class="metric-val" style="color: {'#16A34A' if diff >= 0 else '#DC2626'}; font-size: 1.7rem;">{diff_sign}{diff:.1f}</div>
                    <small>Compared to Cohort Mean</small>
                </div>
                """, unsafe_allow_html=True)

            # Details & Probabilities
            st.markdown("### Letter Grade Probability Breakdown")
            if result['grade_probabilities']:
                prob_df = pd.DataFrame(list(result['grade_probabilities'].items()), columns=['Grade', 'Probability'])
                fig, ax = plt.subplots(figsize=(8, 2.5))
                sns.barplot(data=prob_df, x='Grade', y='Probability', palette='Blues_r', ax=ax)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Probability")
                for p in ax.patches:
                    h = p.get_height()
                    if h > 0.02:
                        ax.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width()/2., h + 0.03), ha='center', fontweight='bold')
                st.pyplot(fig)
                plt.close()

            st.markdown("### Actionable Recommendations")
            for rec in result['actionable_recommendations']:
                st.info(f"{rec}")

    # -------------------------------------------------------------
    # TAB 2: INTERACTIVE EDA
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Cohort Exploratory Data Analysis")
        
        # Filtering Controls
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            selected_grades = st.multiselect("Filter by Grade", options=sorted(df_raw['final_grade'].unique()), default=sorted(df_raw['final_grade'].unique()))
        with f_col2:
            selected_job = st.multiselect("Filter by Part-Time Job", options=df_raw['part_time_job'].unique(), default=df_raw['part_time_job'].unique())
        with f_col3:
            selected_net = st.multiselect("Filter by Internet Access", options=df_raw['internet_access'].unique(), default=df_raw['internet_access'].unique())

        filtered_df = df_raw[
            (df_raw['final_grade'].isin(selected_grades)) &
            (df_raw['part_time_job'].isin(selected_job)) &
            (df_raw['internet_access'].isin(selected_net))
        ]

        st.write(f"Showing **{len(filtered_df)}** of {len(df_raw)} student records.")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            st.markdown("#### Score Distribution by Selected Cohort")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_df['final_exam_score'], kde=True, color='#1E40AF', bins=20, ax=ax)
            ax.set_xlabel("Final Exam Score")
            st.pyplot(fig)
            plt.close()

        with eda_col2:
            st.markdown("#### Study Time vs Exam Score (Trendline)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.regplot(data=filtered_df, x='study_time_hours', y='final_exam_score',
                        scatter_kws={'alpha': 0.5, 'color': '#0284C7'}, line_kws={'color': '#DC2626'}, ax=ax)
            ax.set_xlabel("Daily Study Time (Hours)")
            ax.set_ylabel("Final Exam Score")
            st.pyplot(fig)
            plt.close()

        # Categorical Heatmap
        st.markdown("#### Factor Comparison: Parental Education & Employment")
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot_table = df_raw.pivot_table(index='parental_education', columns='part_time_job', values='final_exam_score', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_ylabel("Parental Education")
        ax.set_xlabel("Part-Time Job")
        st.pyplot(fig)
        plt.close()

    # -------------------------------------------------------------
    # TAB 3: MODEL LEADERBOARD & EXPLAINABILITY
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Machine Learning Model Benchmarks & Explainability")

        if metrics_data:
            m_col1, m_col2 = st.columns(2)

            with m_col1:
                st.markdown("#### Regression Models Leaderboard (Target: Score)")
                reg_df = pd.DataFrame(metrics_data['regression_results'])
                st.dataframe(reg_df, use_container_width=True)

            with m_col2:
                st.markdown("#### Classification Models Leaderboard (Target: Grade)")
                clf_df = pd.DataFrame(metrics_data['classification_results'])
                st.dataframe(clf_df, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Most Important Decision Drivers (Global Explainability)")
            feat_df = pd.DataFrame(metrics_data['top_features'])
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues_r', ax=ax)
            ax.set_xlabel("Model Importance Weight")
            st.pyplot(fig)
            plt.close()

        # Static Saved Visuals
        st.markdown("#### Evaluation Visuals")
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            if os.path.exists("reports/figures/07_actual_vs_predicted_residuals.png"):
                st.image("reports/figures/07_actual_vs_predicted_residuals.png", caption="Actual vs Predicted & Residuals", use_container_width=True)
        with vis_col2:
            if os.path.exists("reports/figures/09_confusion_matrix_classification.png"):
                st.image("reports/figures/09_confusion_matrix_classification.png", caption="Classification Confusion Matrix", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: BATCH CSV PREDICTION
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Batch Student Scoring & Automated Advisory")
        st.markdown("Upload a CSV file containing multiple students or evaluate on the test split.")

        uploaded_file = st.file_uploader("Choose a student CSV file", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
        else:
            st.info("No custom file uploaded. Showing sample evaluation on holdout test set:")
            batch_df = df_raw.sample(15, random_state=42).copy()

        if st.button("Score Batch Records", type="primary"):
            scored_rows = []
            for _, row in batch_df.iterrows():
                student_dict = row.to_dict()
                res = predictor.predict_single(student_dict)
                scored_rows.append({
                    "Predicted Score": res['predicted_final_score'],
                    "Predicted Grade": res['predicted_letter_grade'],
                    "At Risk?": "YES" if res['is_at_risk'] else "NO",
                    "Primary Recommendation": res['actionable_recommendations'][0]
                })

            result_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(scored_rows)], axis=1)
            st.success(f"Successfully scored {len(result_df)} student records!")
            st.dataframe(result_df, use_container_width=True)

            # CSV Download
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scored Students CSV",
                data=csv_data,
                file_name="scored_students_output.csv",
                mime="text/csv",
                type="primary"
            )

if __name__ == "__main__":
    main()
