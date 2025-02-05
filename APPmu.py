import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("dataset_cleaned.csv")

# Define new top features based on user input
top_features = [
    "Household Income Level", "Days to Accept", "Days to Qualify", "Service Quality Rating",
    "Competitor Price Sensitivity", "Days to Install Request", "Bundled Service Interest",
    "Discount Availed (INR)", "Signal Strength"
]
X = pd.get_dummies(df[top_features])  # One-hot encode categorical features if needed
y = df["Installed"]

# Scale features for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

# Train and Evaluate Models
model_metrics = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }
    model_metrics[name] = metrics

# Streamlit Single Page Layout
st.title("📊 Machine Learning Dashboard for Service Installation Prediction")

# Section 1: Model Comparison
st.header("🤖 Model Comparison")
st.write("Compare different machine learning models on key evaluation metrics.")

# Display Metrics
metrics_df = pd.DataFrame(model_metrics).T
st.write("### Model Performance Metrics")
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
st.write("The model uses the following top features to make predictions:")

# Random Forest Feature Importance
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_imp_df["Importance"], y=feature_imp_df["Feature"], palette="viridis", ax=ax)
ax.set_title("Top Features by Importance", fontsize=16)
ax.set_xlabel("Feature Importance", fontsize=14)
ax.set_ylabel("Feature", fontsize=14)
st.pyplot(fig)

# Display Table
st.write("### Feature Importance Table")
st.dataframe(feature_imp_df.style.highlight_max(axis=0, color='lightblue'))

# Section 3: Install Prediction
st.header("🔮 Predict Service Installation")
st.write("Provide the following details about the customer:")

# Input Form
household_income = st.selectbox("Household Income Level", options=df["Household Income Level"].unique())
days_to_accept = st.number_input("Days to Accept", min_value=0, max_value=7, value=2)
days_to_qualify = st.number_input("Days to Qualify", min_value=0, max_value=10, value=3)
service_quality = st.number_input("Service Quality Rating", min_value=1, max_value=5, value=3)
competitor_price = st.selectbox("Competitor Price Sensitivity", options=df["Competitor Price Sensitivity"].unique())
days_to_install = st.number_input("Days to Install Request", min_value=0, max_value=24, value=5)
bundled_service = st.selectbox("Bundled Service Interest", options=df["Bundled Service Interest"].unique())
discount_availed = st.number_input("Discount Availed (INR)", min_value=0, max_value=500, value=100)
signal_strength = st.selectbox("Signal Strength", options=df["Signal Strength"].unique())

# Prediction Button
if st.button("Predict Installation"):
    # Prepare input data
    input_data = pd.DataFrame([[
        household_income, days_to_accept, days_to_qualify, service_quality, competitor_price,
        days_to_install, bundled_service, discount_availed, signal_strength
    ]], columns=top_features)
    input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)
    best_model = models["XGBoost"]  # Use the best-performing model
    prediction = best_model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ The customer is likely to install the service!")
    else:
        st.warning("❌ The customer may **not** install the service.")

# Footer
st.info("Built with ❤️ using Streamlit")
