import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load Dataset
@st.cache_resource
def load_data():
    return pd.read_csv("dataset_cleaned.csv")

df = load_data()

# Define updated feature list
top_features = [
    "Household Income Level", "Days to Accept", "Days to Qualify", "Service Quality Rating",
    "Competitor Price Sensitivity", "Bundled Service Interest",
    "Discount Availed (INR)", "Signal Strength", "Marketing Spend (INR)"
]

# Encode categorical features
le = LabelEncoder()
for col in ["Household Income Level", "Competitor Price Sensitivity", "Bundled Service Interest", "Signal Strength"]:
    df[col] = le.fit_transform(df[col])

# Preprocess Data
@st.cache_resource
def preprocess_data(df, top_features):
    X = df[top_features]
    y = df["Installed"]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

(X_train, X_test, y_train, y_test), feature_columns = preprocess_data(df, top_features)

# Train Models
@st.cache_resource
def train_models():
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=4)
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
st.set_page_config(page_title="Service Installation Dashboard", layout="wide")
st.title("📊 Business Insights Dashboard for Service Installation")

# 🎨 Custom Styling
sns.set_style("darkgrid")

# Section 1: Model Comparison
st.header("🤖 Model Comparison")
st.write("Compare different machine learning models on key evaluation metrics.")

metrics_df = pd.DataFrame(model_metrics).T
st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))

fig, ax = plt.subplots(figsize=(8, 6))
metrics_df.plot(kind="bar", ax=ax, colormap="coolwarm")
ax.set_title("Model Performance Comparison", fontsize=16)
ax.set_ylabel("Score", fontsize=14)
st.pyplot(fig)

# Section 2: Dynamic Data Exploration
st.header("📊 Data Exploration & Analysis")

# 📌 Select Feature for Dynamic Graph
feature_selection = st.selectbox("Select Feature for Analysis:", ["Household Income Level", "Service Quality Rating", "Marketing Channel"])
if feature_selection == "Household Income Level":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df["Household Income Level"], y=df["Installed"], palette="viridis", ax=ax)
    ax.set_title("Installation Rate by Household Income Level")
    st.pyplot(fig)

elif feature_selection == "Service Quality Rating":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["Installed"], y=df["Service Quality Rating"], palette="coolwarm", ax=ax)
    ax.set_title("Service Quality Rating vs Installation")
    st.pyplot(fig)

elif feature_selection == "Marketing Channel":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df["Marketing Channel"], y=df["Installed"], palette="coolwarm", ax=ax)
    ax.set_title("Conversion Rate by Marketing Channel")
    st.pyplot(fig)

# Section 3: Seasonal Trends in Installations
st.header("📅 Seasonal Trends in Installations")
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=df["Lead Created Month"], y=df["Installed"].rolling(5).mean(), marker="o", color="red", ax=ax)
ax.set_title("Installations Over Time", fontsize=16)
st.pyplot(fig)

# Section 4: Install Prediction
st.header("🔮 Predict Service Installation")
household_income = st.selectbox("Household Income Level", df["Household Income Level"].unique())
days_to_accept = st.slider("Days to Accept", 0, 7, 2)
days_to_qualify = st.slider("Days to Qualify", 0, 10, 3)
service_quality = st.slider("Service Quality Rating", 1, 5, 3)
competitor_price = st.selectbox("Competitor Price Sensitivity", df["Competitor Price Sensitivity"].unique())
bundled_service = st.selectbox("Bundled Service Interest", df["Bundled Service Interest"].unique())
discount_availed = st.slider("Discount Availed (INR)", 0, 500, 100)
signal_strength = st.selectbox("Signal Strength", df["Signal Strength"].unique())
marketing_spend = st.slider("Marketing Spend (INR)", 5000, 20000, 7749)

if st.button("Predict Installation"):
    input_data = pd.DataFrame([[
        household_income, days_to_accept, days_to_qualify, service_quality, competitor_price,
        bundled_service, discount_availed, signal_strength, marketing_spend
    ]], columns=top_features)

    best_model = trained_models["XGBoost"]
    prediction = best_model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ The customer is likely to install the service!")
    else:
        st.warning("❌ The customer may **not** install the service!")

st.info("Built with ❤️ using Streamlit")
