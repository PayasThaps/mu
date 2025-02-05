import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load Dataset
df = pd.read_csv("dataset_cleaned.csv")

# Define top 10 features based on importance
top_features = [
    "Days to Install Request", "Days to Qualify", "Days to Accept", "Lifetime Value (INR)",
    "Marketing Spend (INR)", "Discount Availed (INR)", "Time Spent on Research (Days)",
    "Distance to Service Hub", "Lead Created Day", "Network Downtime (Hours)"
]
X = df[top_features]
y = df["Installed"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Train and Evaluate Models
model_metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }
    model_metrics[name] = metrics

# Sidebar
st.sidebar.title("📊 Explore Dashboard")
page = st.sidebar.radio("Choose a section", ["Model Comparison", "Feature Importance", "Install Prediction"])

# Model Comparison Section
if page == "Model Comparison":
    st.title("🤖 Model Comparison")
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

# Feature Importance Section
elif page == "Feature Importance":
    st.title("🔬 Feature Importance in Random Forest")
    st.write("The model uses the following top 10 features to make predictions:")

    # Random Forest Feature Importance
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({"Feature": top_features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_imp_df["Importance"], y=feature_imp_df["Feature"], palette="viridis", ax=ax)
    ax.set_title("Top 10 Features by Importance", fontsize=16)
    ax.set_xlabel("Feature Importance", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    st.pyplot(fig)

    # Display Table
    st.write("### Feature Importance Table")
    st.dataframe(feature_imp_df.style.highlight_max(axis=0, color='lightblue'))

# Install Prediction Section
elif page == "Install Prediction":
    st.title("🔮 Predict Service Installation")
    st.write("Provide the following details about the customer:")

    # Input Form
    days_to_install = st.number_input("Days to Install Request", min_value=0, max_value=24, value=5)
    days_to_qualify = st.number_input("Days to Qualify", min_value=0, max_value=10, value=3)
    days_to_accept = st.number_input("Days to Accept", min_value=0, max_value=7, value=2)
    lifetime_value = st.number_input("Lifetime Value (INR)", min_value=999, max_value=75564, value=29123)
    marketing_spend = st.number_input("Marketing Spend (INR)", min_value=5000, max_value=20000, value=7749)
    discount_availed = st.number_input("Discount Availed (INR)", min_value=0, max_value=500, value=100)
    time_spent_research = st.number_input("Time Spent on Research (Days)", min_value=1, max_value=30, value=15)
    distance_service_hub = st.number_input("Distance to Service Hub (km)", min_value=5, max_value=50, value=20)
    lead_created_day = st.number_input("Lead Created Day", min_value=1, max_value=31, value=15)
    network_downtime = st.number_input("Network Downtime (Hours)", min_value=0, max_value=24, value=6)

    # Prediction Button
    if st.button("Predict Installation"):
        input_data = np.array([[days_to_install, days_to_qualify, days_to_accept, lifetime_value,
                                marketing_spend, discount_availed, time_spent_research,
                                distance_service_hub, lead_created_day, network_downtime]])
        best_model = models["XGBoost"]  # Use the best-performing model
        prediction = best_model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ The customer is likely to install the service!")
        else:
            st.warning("❌ The customer may **not** install the service.")

# Footer
st.sidebar.info("Built with ❤️ using Streamlit")
