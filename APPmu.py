import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("dataset_cleaned.csv")

# Train Random Forest globally (to be used in multiple sections)
features = ["Plan Cost (INR)", "Average Monthly Spend (INR)", "Lifetime Value (INR)", "Marketing Spend (INR)"]
X = df[features]
y = df["Installed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Sidebar
st.sidebar.title("🔍 Data Exploration & Model Prediction")
page = st.sidebar.radio("Choose a section", ["Data Overview", "Feature Importance", "Install Prediction"])

# Data Overview
if page == "Data Overview":
    st.title("📊 Dataset Overview")
    st.write("#### Sample Data:")
    st.dataframe(df.head())

    st.write("#### Categorical Data Distribution:")
    categorical_cols = ["Plan Type", "Household Income Level", "Lead Source", "Marketing Channel"]
    for col in categorical_cols:
        st.subheader(f"📌 {col}")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Feature Importance
elif page == "Feature Importance":
    st.title("📌 Feature Importance in Install Prediction")

    # Get feature importance
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot
    st.subheader("🔬 Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=feature_imp_df["Feature"], y=feature_imp_df["Importance"], palette="coolwarm")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Display Table
    st.dataframe(feature_imp_df)

# Install Prediction
elif page == "Install Prediction":
    st.title("🔮 Predict Service Installation")

    # Input Form
    st.subheader("📌 Enter Customer Details:")
    plan_cost = st.number_input("Plan Cost (INR)", min_value=799, max_value=1599, value=1299)
    monthly_spend = st.number_input("Average Monthly Spend (INR)", min_value=999, max_value=2099, value=1580)
    lifetime_value = st.number_input("Lifetime Value (INR)", min_value=999, max_value=75564, value=29123)
    marketing_spend = st.number_input("Marketing Spend (INR)", min_value=5000, max_value=20000, value=7749)

    # Prediction Button
    if st.button("Predict Installation"):
        input_data = np.array([[plan_cost, monthly_spend, lifetime_value, marketing_spend]])
        prediction = rf_model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ This customer is likely to install the service!")
        else:
            st.warning("❌ This customer may **not** install the service.")

# Footer
st.sidebar.info("Built with ❤️ using Streamlit")
