import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Logistic Regression Example",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# Pink UI Theme (INLINE CSS)
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fbc2eb, #fda4af);
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #831843;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}
h2, h3 {
    color: #9d174d;
}
.card {
    background-color: #fff1f2;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: 24px;
}
div[data-testid="metric-container"] {
    background-color: #fce7f3;
    border-radius: 14px;
    padding: 16px;
}
.stButton > button {
    background-color: #db2777;
    color: white;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #be185d;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<h1>Logistic Regression Example</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Telecom Customer Churn Dataset</h3>", unsafe_allow_html=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (2).csv")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------------------------
# Train Model
# -------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Model Performance
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
with c2:
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.2f}")
with c3:
    st.metric("Test Samples", X_test.shape[0])

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Confusion Matrix (GRAPH – PLOTLY, SAFE)
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted No", "Predicted Yes"],
    y=["Actual No", "Actual Yes"],
    colorscale="RdPu",
    showscale=True
)

fig.update_layout(
    width=420,
    height=420,
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
with st.expander("📄 View Classification Report"):
    st.text(classification_report(y_test, y_pred))

# -------------------------------------------------
# Predict New Customer
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Predict Churn for a New Customer")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

input_df = pd.DataFrame([[
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    internet_map[internet],
    payment_map[payment]
]], columns=[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod"
])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f" Customer is likely to CHURN (Probability: {prob*100:.2f}%)")
    else:
        st.success(f" Customer is likely to STAY (Probability: {(1-prob)*100:.2f}%)")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Business Interpretation
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Business Interpretation")
st.write(
    "It is better to wrongly flag a loyal customer as churn than to miss a customer "
    "who is actually going to leave, because customer churn directly affects revenue."
)
st.markdown("</div>", unsafe_allow_html=True)
