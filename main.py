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


st.markdown("""
<style>
.big-title {
    font-size: 30px;
    font-weight: bold;
}
.section {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)
def main():
    st.set_page_config(layout="wide")

    # =========================
    # HEADER
    # =========================
    st.markdown('<div class="big-title">🏦 Personal Loan Approval System</div>', unsafe_allow_html=True)
    st.markdown("Simple AI system to predict loan approval using SVM, KNN, and ANN")

    st.divider()

    df, le_dict = load_data()

    X = df.drop("Loan_Approval_Status", axis=1)
    y = df["Loan_Approval_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0]

    st.success(f"🏆 Best Model: {best_model}")

    st.divider()

    # =========================
    # DASHBOARD
    # =========================
    st.markdown("### 📊 Model Performance")
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["SVM", "KNN", "ANN"])

    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            res = results[name]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{res['precision']:.2f}")
            col3.metric("Recall", f"{res['recall']:.2f}")
            col4.metric("F1 Score", f"{res['f1']:.2f}")

            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    st.divider()

    # =========================
    # INPUT SECTION
    # =========================
    st.markdown("### 📝 Applicant Information")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        with st.form("form"):
            cols = st.columns(3)
            data = {}

            for i, col in enumerate(X.columns):
                if col == "Debt_Income_Ratio":
                    continue

                with cols[i % 3]:
                    if col in le_dict:
                        val = st.selectbox(col, le_dict[col].classes_)
                        data[col] = le_dict[col].transform([val])[0]
                    else:
                        data[col] = st.number_input(col, value=int(df[col].mean()), step=1)

            selected_model = st.selectbox("Select Model", list(models.keys()))

            submit = st.form_submit_button("🚀 Predict")

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

        st.divider()

        # =========================
        # RESULT SECTION
        # =========================
        st.markdown("### 🔍 Prediction Result")

        c1, c2 = st.columns(2)
        c1.metric("Approval Probability", f"{prob*100:.2f}%")
        c2.progress(float(prob))

        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # =========================
        # RISK
        # =========================
        st.markdown("### ⚠️ Risk Score")
        r = risk_score(input_df)
        st.metric("Risk Level", f"{r}/100")

        # =========================
        # COMPARE MODELS
        # =========================
        st.markdown("### 🔄 Model Comparison")

        comparison = {}
        for m in models:
            comparison[m] = models[m].predict_proba(scaled)[0][1]

        st.bar_chart(pd.DataFrame.from_dict(comparison, orient="index"))

        # =========================
        # EXPLANATION
        # =========================
        st.markdown("### 🧠 Explanation")

        reasons = []

        if input_df["Credit_Score"].iloc[0] < 600:
            reasons.append("⚠️ Low credit score")

        if input_df["Debt_Income_Ratio"].iloc[0] > 0.4:
            reasons.append("⚠️ High debt ratio")

        if input_df["Existing_Loans"].iloc[0] > 3:
            reasons.append("⚠️ Too many loans")

        if input_df["Annual_Income"].iloc[0] > 60000:
            reasons.append("✅ Strong income")

        if input_df["Loan_History"].iloc[0] == 1:
            reasons.append("✅ Good history")

        for r in reasons:
            st.write(r)

        if pred == 1:
            st.info("📌 Low risk applicant")
        else:
            st.info("📌 High risk applicant")
