import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

# ─── Train and save model if not exists ───
def train_model():
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "tenure_months": np.random.randint(1, 72, n),
        "num_products": np.random.randint(1, 5, n),
        "has_internet": np.random.randint(0, 2, n),
        "has_phone": np.random.randint(0, 2, n),
        "support_calls": np.random.randint(0, 10, n),
        "payment_delay_days": np.random.randint(0, 30, n),
    })

    # Churn logic
    churn_score = (
        (data["monthly_charges"] > 80).astype(int) * 2 +
        (data["tenure_months"] < 12).astype(int) * 2 +
        (data["support_calls"] > 5).astype(int) * 2 +
        (data["payment_delay_days"] > 15).astype(int) +
        (data["num_products"] == 1).astype(int)
    )
    data["churn"] = (churn_score >= 4).astype(int)

    X = data.drop("churn", axis=1)
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")

    return model, list(X.columns), accuracy

# ─── Load or train model ───
@st.cache_resource
def load_model():
    if os.path.exists("churn_model.pkl"):
        model = joblib.load("churn_model.pkl")
        features = joblib.load("feature_names.pkl")
        return model, features, None
    else:
        return train_model()

# ─── Gauge chart ───
def make_gauge(probability):
    color = "#2ecc71" if probability < 0.4 else "#f39c12" if probability < 0.7 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Risk %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#d5f5e3"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
        }
    ))
    fig.update_layout(height=300)
    return fig

# ─── Feature importance chart ───
def make_feature_chart(model, features):
    importance = model.feature_importances_
    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        title="What Drives Churn?",
        labels={"x": "Importance", "y": "Feature"},
        color=importance,
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=350, showlegend=False)
    return fig

# ─── Main App ───
def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

    st.title("📊 Customer Churn Prediction System")
    st.write("Enter customer details to predict whether they are likely to churn.")

    model, features, accuracy = load_model()

    if accuracy:
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.1%}")

    st.markdown("---")

    # ─── Input Form ───
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Info")
        age = st.slider("Age", 18, 70, 35)
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_internet = st.selectbox("Has Internet Service?", ["Yes", "No"])
        has_phone = st.selectbox("Has Phone Service?", ["Yes", "No"])

    with col2:
        st.subheader("💳 Billing Info")
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
        support_calls = st.slider("Support Calls (last 6 months)", 0, 10, 2)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)

    st.markdown("---")

    if st.button("🔮 Predict Churn", use_container_width=True):
        input_data = pd.DataFrame([{
            "age": age,
            "monthly_charges": monthly_charges,
            "tenure_months": tenure,
            "num_products": num_products,
            "has_internet": 1 if has_internet == "Yes" else 0,
            "has_phone": 1 if has_phone == "Yes" else 0,
            "support_calls": support_calls,
            "payment_delay_days": payment_delay,
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH CHURN RISK — This customer is likely to leave!")
            else:
                st.success("✅ LOW CHURN RISK — This customer is likely to stay!")

            st.plotly_chart(make_gauge(probability), use_container_width=True)

        with col2:
            st.plotly_chart(make_feature_chart(model, features), use_container_width=True)

        st.markdown("### 📋 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{probability:.1%}")
        col2.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
        col3.metric("Risk Level", "High 🔴" if probability > 0.7 else "Medium 🟡" if probability > 0.4 else "Low 🟢")

if __name__ == "__main__":
    main()
