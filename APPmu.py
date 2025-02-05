import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load Dataset
@st.cache_resource
def load_data():
    return pd.read_csv("dataset_cleaned.csv")

df = load_data()

# Define updated top features (Removed "Days to Install Request")
top_features = [
    "Household Income Level", "Days to Accept", "Days to Qualify", "Service Quality Rating",
    "Competitor Price Sensitivity", "Bundled Service Interest",
    "Discount Availed (INR)", "Signal Strength"
]

# Preprocess Data
@st.cache_resource
def preprocess_data(df, top_features):
    X = pd.get_dummies(df[top_features])
    y = df["Installed"]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

(X_train, X_test, y_train, y_test), feature_columns = preprocess_data(df, top_features)

# Train Models
@st.cache_resource
def train_models():
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }
    trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}
    return trained_models

trained_models = train_models()

# Evaluate Models
@st.cache_resource
def evaluate_models():
    metrics = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        }
    return metrics

model_metrics = evaluate_models()

# Streamlit Layout
st.title("📊 Optimized Service Installation Prediction Dashboard")

# Section 1: Model Comparison
st.header("🤖 Model Comparison")
st.write("Compare different machine learning models on key evaluation metrics.")
metrics_df = pd.DataFrame(model_metrics).T
st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))

# Visualization
st.write("### Model Comparison Chart")
fig, ax = plt.subplots(figsize=(8, 6))
metrics_df.plot(kind="bar", ax=ax)
ax.set_title("Model Performance Comparison", fontsize=16)
ax.set_ylabel("Score", fontsize=14)
ax.set_xticklabels(metrics_df.index, rotation=45, fontsize=12)
st.pyplot(fig)

# Section 2: Feature Importance
st.header("🔬 Feature Importance in Random Forest")
rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({"Feature": feature_columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_imp_df["Importance"], y=feature_imp_df["Feature"], palette="viridis", ax=ax)
ax.set_title("Top Features by Importance", fontsize=16)
st.pyplot(fig)

# Section 3: Install Prediction
st.header("🔮 Predict Service Installation")
household_income = st.selectbox("Household Income Level", options=df["Household Income Level"].unique())
days_to_accept = st.number_input("Days to Accept", min_value=0, max_value=7, value=2)
days_to_qualify = st.number_input("Days to Qualify", min_value=0, max_value=10, value=3)
service_quality = st.number_input("Service Quality Rating", min_value=1, max_value=5, value=3)
competitor_price = st.selectbox("Competitor Price Sensitivity", options=df["Competitor Price Sensitivity"].unique())
bundled_service = st.selectbox("Bundled Service Interest", options=df["Bundled Service Interest"].unique())
discount_availed = st.number_input("Discount Availed (INR)", min_value=0, max_value=500, value=100)
signal_strength = st.selectbox("Signal Strength", options=df["Signal Strength"].unique())

if st.button("Predict Installation"):
    input_data = pd.DataFrame([[
        household_income, days_to_accept, days_to_qualify, service_quality, competitor_price,
        bundled_service, discount_availed, signal_strength
    ]], columns=top_features)
    input_data = pd.get_dummies(input_data).reindex(columns=feature_columns, fill_value=0)
    best_model = trained_models["XGBoost"]
    prediction = best_model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ The customer is likely to install the service!")
    else:
        st.warning("❌ The customer may **not** install the service!")

st.info("Built with ❤️ using Streamlit")
