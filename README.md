# 📊 Customer Churn Prediction System

An end-to-end machine learning web app that predicts whether a customer is likely to leave a business — built with scikit-learn, Streamlit, and Plotly.

## 🚀 Features
- 🔮 Real-time churn prediction with probability score
- 📊 Visual churn risk gauge (Low / Medium / High)
- 📈 Feature importance chart — shows exactly what drives churn
- 🎛️ Interactive sliders for customer data input
- 🤖 Random Forest model trained on realistic customer data
- 📋 Clean summary dashboard with key metrics

## 🛠️ Tech Stack
- Python
- Scikit-learn (Random Forest Classifier)
- Streamlit
- Plotly
- Pandas
- Joblib

## 🧠 How the ML Model Works
1. **Data** — Trained on 1000 synthetic customer records
2. **Features** — Age, monthly charges, tenure, products, support calls, payment delays
3. **Algorithm** — Random Forest with 100 decision trees
4. **Output** — Churn probability (0-100%) + binary prediction
5. **Accuracy** — ~85%+ on test data

## 📊 What Drives Customer Churn?
Based on the model's feature importance:
- 💰 **Monthly Charges** — highest billing = higher churn risk
- 📞 **Support Calls** — frustrated customers call more before leaving
- 📅 **Tenure** — newer customers churn more than loyal ones
- ⏰ **Payment Delays** — late payments signal dissatisfaction

## ⚙️ How It Works
1. Enter customer details using the sliders
2. Click "Predict Churn"
3. See the churn risk gauge, probability, and feature importance
4. Use insights to take action before the customer leaves

## 📦 Installation
```bash
git clone https://github.com/RanaQadeer/ml-prediction-api.git
cd ml-prediction-api
pip install -r requirements.txt
python -m streamlit run app.py
```

## 💼 Business Use Cases
- **Telecom companies** — identify subscribers about to cancel
- **SaaS businesses** — flag users likely to stop their subscription
- **Banks** — detect customers moving to competitors
- **E-commerce** — find shoppers who won't return
- **Insurance** — predict policy non-renewals

## 💰 ROI for Businesses
Acquiring a new customer costs 5-7x more than retaining one.
A churn prediction system that saves even 10 customers/month
at $100 average value = $1,000/month saved = $12,000/year.

## 📫 Want This for Your Business?
Contact me on Fiverr: [your-fiverr-link]
