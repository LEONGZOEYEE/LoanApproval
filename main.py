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

    df["Debt_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)
    
    for col in df.columns:
        # Try converting to numeric if possible
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    le_dict = {}
    for col in df.columns:
        if df[col].dtype == "object":
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
    st.set_page_config(layout="wide")

    st.title("🏦 Personal Loan Approval Prediction System")
    st.caption("Smart AI-based loan approval decision system")

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

    # =========================
    # MODEL TABS
    # =========================
    st.subheader("📊 Model Dashboard")

    tabs = st.tabs(["🟢 SVM", "🔵 KNN", "🟣 ANN"])

    # =========================
    # EACH TAB
    # =========================
    for i, name in enumerate(["SVM", "KNN", "ANN"]):
        with tabs[i]:
            model = models[name]
            res = results[name]

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            c2.metric("Precision", f"{res['precision']:.2f}")
            c3.metric("Recall", f"{res['recall']:.2f}")
            c4.metric("F1", f"{res['f1']:.2f}")

            # Confusion Matrix
            fig, ax = plt.subplots()
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    # =========================
    # FORM INPUT (GLOBAL)
    # =========================
    st.subheader("🔧 Applicant Information")

    with st.form("form"):
        data = {}

        for col in X.columns:
            if col == "Debt_Income_Ratio":
                continue
            elif col in le_dict:
                val = st.selectbox(col, le_dict[col].classes_)
                data[col] = le_dict[col].transform([val])[0]
            else:
                data[col] = st.number_input(col, value=int(df[col].mean()), step=1)

        # MODEL SELECTION
        selected_model = st.selectbox("Select Model", list(models.keys()))

        submit = st.form_submit_button("🚀 Predict")

    input_df = pd.DataFrame([data])

    # auto compute ratio (IMPORTANT)
    input_df["Debt_Income_Ratio"] = (
        input_df["Outstanding_Debt"] / (input_df["Annual_Income"] + 1)
    )

    # =========================
    # PREDICTION
    # =========================
    if submit:
        # recreate input
        input_df = pd.DataFrame([data])

        # ✅ ALWAYS recompute ratio AFTER creating dataframe
        input_df["Debt_Income_Ratio"] = (
            input_df["Outstanding_Debt"] / (input_df["Annual_Income"] + 1)
        )

        st.write(f"Calculated Debt-Income Ratio: {input_df['Debt_Income_Ratio'].iloc[0]:.2f}")

        # ✅ ensure column order matches training
        input_df = input_df[X.columns]

        scaled = scaler.transform(input_df)

        model = models[selected_model]

        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]
        
        st.subheader("🔍 Prediction Result")

        st.metric("Approval Probability", f"{prob*100:.2f}%")
        st.progress(float(prob))

        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # =========================
        # RISK
        # =========================
        st.subheader("⚠️ Risk Score")

        r = risk_score(input_df)
        st.metric("Risk Level", f"{r}/100")

        # =========================
        # COMPARE MODELS
        # =========================
        st.subheader("🔄 Compare Models")

        comparison = {}
        for m in models:
            comparison[m] = models[m].predict_proba(scaled)[0][1]

        st.bar_chart(pd.DataFrame.from_dict(comparison, orient="index"))

        # =========================
        # EXPLANATION
        # =========================
        st.subheader("🧠 Explanation")

        reasons = []

        # Key rules based on domain knowledge
        if input_df["Credit_Score"].iloc[0] < 600:
            reasons.append("⚠️ Low credit score increases risk")

        if input_df["Debt_Income_Ratio"].iloc[0] > 0.4:
            reasons.append("⚠️ High debt-to-income ratio")

        if input_df["Existing_Loans"].iloc[0] > 3:
            reasons.append("⚠️ Too many existing loans")

        if input_df["Annual_Income"].iloc[0] > 60000:
            reasons.append("✅ Strong income supports approval")

        if input_df["Loan_History"].iloc[0] == 1:
            reasons.append("✅ Good loan repayment history")

        # Show results
        if len(reasons) == 0:
            st.write("✅ Applicant has balanced financial profile")
        else:
            for r in reasons:
                st.write(r)

        if pred == 1:
            st.info("📌 Model decision: Applicant is likely safe for loan approval")
        else:
            st.info("📌 Model decision: Applicant is considered risky")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
