import streamlit as st
import pandas as np
import pandas as pd
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

# Define features for modeling
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
st.title("📊 Business Insights Dashboard for Service Installation")

# Section 1: Customer Segmentation Analysis
st.header("🔍 Customer Segmentation Analysis")
segmentation_df = df.groupby("Household Income Level")["Installed"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Household Income Level", y="Installed", data=segmentation_df, palette="coolwarm", ax=ax)
ax.set_title("Installation Rate by Household Income Level")
ax.set_ylabel("Installation Rate")
st.pyplot(fig)

# Section 2: Churn Prediction Analysis
st.header("🚨 Churn Prediction Analysis")
st.write("What factors contribute to customers **not installing** the service?")
churn_df = df.groupby("Competitor Price Sensitivity")["Installed"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Competitor Price Sensitivity", y="Installed", data=churn_df, palette="coolwarm", ax=ax)
ax.set_title("Installation Rate by Competitor Price Sensitivity")
st.pyplot(fig)

# Section 3: Impact of Discounts on Installations
st.header("💰 Discount Impact on Installations")
st.write("Are discounts leading to higher installation rates?")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df["Installed"], y=df["Discount Availed (INR)"], palette="coolwarm", ax=ax)
ax.set_title("Discounts Availed vs. Installation")
st.pyplot(fig)

# Section 4: Service Quality vs Installations
st.header("📡 Service Quality & Signal Strength Impact")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df["Installed"], y=df["Service Quality Rating"], palette="coolwarm", ax=ax)
ax.set_title("Service Quality Rating & Installation")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df["Signal Strength"], hue=df["Installed"], palette="coolwarm", ax=ax)
ax.set_title("Signal Strength & Installation Rates")
st.pyplot(fig)

# Section 5: Marketing Channel Performance
st.header("📢 Marketing Channel Effectiveness")
st.write("Which marketing channels are performing the best?")
marketing_df = df.groupby("Marketing Channel")["Installed"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Marketing Channel", y="Installed", data=marketing_df, palette="coolwarm", ax=ax)
ax.set_title("Conversion Rate by Marketing Channel")
st.pyplot(fig)

# Section 6: Seasonal Trends in Installations
st.header("📅 Seasonal Trends in Installations")
st.write("How do installations change over time?")
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=df["Lead Created Month"], y=df["Installed"].rolling(5).mean(), marker="o", ax=ax)
ax.set_title("Installations Over Time")
st.pyplot(fig)

# Section 7: Install Prediction
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
