import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Loan Dataset.csv", index_col=0)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fix numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    # Fix categorical columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Create feature AFTER cleaning
    df["Debt_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)

    # Encode categorical
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

# =========================
# TRAIN
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    for m in models:
        models[m].fit(X_train, y_train)

    return models

# =========================
# EVALUATE
# =========================
def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred)
        }

    return results

# =========================
# RISK SCORE
# =========================
def risk_score(df):
    score = 0

    if df["Credit_Score"].iloc[0] < 600:
        score += 30
    if df["Debt_Income_Ratio"].iloc[0] > 0.4:
        score += 30
    if df["Annual_Income"].iloc[0] < 30000:
        score += 20
    if df["Existing_Loans"].iloc[0] > 3:
        score += 20

    return min(score, 100)

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(
        page_title="Personal Loan Approval System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # =========================
    # UI STYLE
    # =========================
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 34px;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .sub-title {
                font-size: 16px;
                color: #666666;
                margin-top: 0;
                margin-bottom: 1rem;
            }
            .section-box {
                background: #ffffff;
                padding: 18px;
                border-radius: 12px;
                border: 1px solid #e6e6e6;
                margin-bottom: 18px;
            }
            .small-note {
                color: #666666;
                font-size: 13px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # HEADER
    # =========================
    st.markdown('<div class="main-title">🏦 Personal Loan Approval Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simple AI-based system to predict loan approval using SVM, KNN, and ANN</div>', unsafe_allow_html=True)

    df, le_dict = load_data()

    X = df.drop("Loan_Approval_Status", axis=1)
    y = df["Loan_Approval_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0]

    # =========================
    # SIDEBAR
    # =========================
    with st.sidebar:
        st.markdown("## 📌 System Summary")
        st.info(f"Best Model: **{best_model}**")
        st.write("This system compares:")
        st.write("- Support Vector Machine (SVM)")
        st.write("- K-Nearest Neighbors (KNN)")
        st.write("- Artificial Neural Network (ANN)")

        st.markdown("## 📋 How to Use")
        st.write("1. Check model performance")
        st.write("2. Fill in applicant information")
        st.write("3. Select a model")
        st.write("4. Click Predict")

        st.markdown("## ℹ️ Note")
        st.caption("The form below uses the same trained models and evaluation logic from your code.")

    st.success(f"🏆 Best Model: {best_model}")

    st.markdown("---")

    # =========================
    # MODEL DASHBOARD
    # =========================
    st.markdown("## 📊 Model Performance")

    tabs = st.tabs(["SVM", "KNN", "ANN"])

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]

            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            c2.metric("Precision", f"{res['precision']:.2f}")
            c3.metric("Recall", f"{res['recall']:.2f}")
            c4.metric("F1 Score", f"{res['f1']:.2f}")

            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # INPUT SECTION
    # =========================
    st.markdown("## 📝 Applicant Information")

    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        with st.form("form"):
            st.markdown("### Fill in the applicant details")

            data = {}
            field_list = [col for col in X.columns if col != "Debt_Income_Ratio"]

            # split fields into 3 neat columns
            cols = st.columns(3)

            for i, col in enumerate(field_list):
                with cols[i % 3]:
                    if col in le_dict:
                        val = st.selectbox(col, le_dict[col].classes_, key=f"input_{col}")
                        data[col] = le_dict[col].transform([val])[0]
                    else:
                        data[col] = st.number_input(
                            col,
                            value=float(df[col].mean()),
                            step=1.0,
                            key=f"input_{col}"
                        )

            st.markdown("### Select Model")
            selected_model = st.selectbox("Model", list(models.keys()))

            submit = st.form_submit_button("🚀 Predict")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # PREDICTION
    # =========================
    if submit:
        input_df = pd.DataFrame([data])

        input_df["Debt_Income_Ratio"] = (
            input_df["Outstanding_Debt"] / (input_df["Annual_Income"] + 1)
        )

        input_df = input_df[X.columns]
        scaled = scaler.transform(input_df)

        model = models[selected_model]

        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]

        st.markdown("---")
        st.markdown("## 🔍 Prediction Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Approval Probability", f"{prob*100:.2f}%")
            st.metric("Risk Level", f"{risk_score(input_df)}/100")

            if pred == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

        with col2:
            st.progress(float(prob))
            st.write(f"Calculated Debt-Income Ratio: **{input_df['Debt_Income_Ratio'].iloc[0]:.2f}**")

        # =========================
        # COMPARE MODELS
        # =========================
        st.markdown("## 🔄 Model Comparison")

        comparison = {}
        for m in models:
            comparison[m] = models[m].predict_proba(scaled)[0][1]

        st.bar_chart(pd.DataFrame.from_dict(comparison, orient="index"))

        # =========================
        # EXPLANATION
        # =========================
        with st.expander("🧠 View Explanation", expanded=True):
            reasons = []

            if input_df["Credit_Score"].iloc[0] < 600:
                reasons.append("⚠️ Low credit score")

            if input_df["Debt_Income_Ratio"].iloc[0] > 0.4:
                reasons.append("⚠️ High debt-to-income ratio")

            if input_df["Existing_Loans"].iloc[0] > 3:
                reasons.append("⚠️ Too many existing loans")

            if input_df["Annual_Income"].iloc[0] > 60000:
                reasons.append("✅ Strong income supports approval")

            if input_df["Loan_History"].iloc[0] == 1:
                reasons.append("✅ Good loan repayment history")

            if len(reasons) == 0:
                st.write("✅ Applicant has a balanced financial profile")
            else:
                for reason in reasons:
                    st.write(reason)

            if pred == 1:
                st.info("📌 Model decision: Applicant is likely safe for loan approval")
            else:
                st.info("📌 Model decision: Applicant is considered risky")

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
