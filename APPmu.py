import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("dataset_cleaned.csv")

# Train Random Forest globally with top 10 features
top_features = [
    "Days to Install Request", "Days to Qualify", "Days to Accept", "Lifetime Value (INR)",
    "Marketing Spend (INR)", "Discount Availed (INR)", "Time Spent on Research (Days)",
    "Distance to Service Hub", "Lead Created Day", "Network Downtime (Hours)"
]
X = df[top_features]
y = df["Installed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Sidebar
st.sidebar.title("📊 Explore Dashboard")
page = st.sidebar.radio("Choose a section", ["Feature Importance", "Install Prediction"])

# Feature Importance Section
if page == "Feature Importance":
    st.title("🔬 Feature Importance in Install Prediction")
    st.write("The model uses the following top 10 features to make predictions:")

    # Get feature importance
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({"Feature": top_features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_imp_df["Importance"], y=feature_imp_df["Feature"], palette="viridis", ax=ax)
    ax.set_title("Top 10 Features by Importance", fontsize=16)
    ax.set_xlabel("Feature Importance", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    st.pyplot(fig)

    # Display Feature Importance Table
    st.write("### Feature Importance Table")
    st.dataframe(feature_imp_df.style.highlight_max(axis=0, color='lightblue'))

# Install Prediction Section
elif page == "Install Prediction":
    st.title("🔮 Predict Service Installation")

    # Input Form
    st.write("Provide the following details about the customer:")
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
        prediction = rf_model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ The customer is likely to install the service!")
        else:
            st.warning("❌ The customer may **not** install the service.")

# Footer
st.sidebar.info("Built with ❤️ using Streamlit")
