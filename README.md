# 📊 Customer Churn Prediction & Explainable AI Dashboard

An end-to-end **Machine Learning and Data Science project** that predicts whether a telecom customer is likely to churn. The project combines **Logistic Regression, data analysis, SHAP explainability, and Streamlit** to provide both predictions and understandable reasons behind them.

## 🚀 Project Overview

Customer churn is an important business problem for telecom companies. This project analyzes customer demographic, service, and account information to estimate the probability of customer churn.

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature encoding
* Machine Learning model training
* Model comparison
* Logistic Regression
* Model evaluation
* SHAP-based Explainable AI
* Streamlit dashboard
* Churn risk classification
* Business recommendations

---

## 🧠 Machine Learning Approach

The project follows this workflow:

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Preprocessing
       ↓
Train/Test Split
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Best Model: Logistic Regression
       ↓
SHAP Explainability
       ↓
Streamlit Dashboard
```

---

## 📂 Dataset

The project uses the **Telco Customer Churn Dataset**, containing customer demographic, service, and account information.

Important features include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

**Target variable:**

```text
Churn
```

where the model predicts whether a customer will leave the service.

---

## 🤖 Model

Multiple machine learning models were evaluated during the project.

The **Logistic Regression model** was selected as the best-performing model based on the evaluation results of this project.

The trained model is saved using:

```python
joblib
```

and stored in:

```text
models/churn_model.pkl
```

---

## 🔍 Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand individual predictions.

Instead of only saying:

> "This customer has a high churn probability."

the application can show which features contributed to that prediction.

For example:

```text
Feature              SHAP Value
--------------------------------
Month-to-month       +0.82
Fiber optic          +0.51
Monthly Charges      +0.34
Tenure               -0.27
Online Security      -0.19
```

### Interpretation

* **Positive SHAP value** → pushes the prediction toward higher churn.
* **Negative SHAP value** → pushes the prediction toward lower churn.
* Larger absolute SHAP values indicate stronger influence on the model output.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

Users can enter customer information such as:

* Customer demographics
* Tenure
* Internet service
* Online services
* Contract type
* Payment method
* Monthly charges
* Total charges

The dashboard provides:

### 🎯 Churn Prediction

```text
Churn Probability: 72.43%

Risk Level: High Risk
```

### 📈 Prediction Explanation

The dashboard displays:

* Top factors influencing the prediction
* SHAP values
* Feature contribution graph
* Feature importance table

### 💡 Business Recommendation

Based on the predicted churn probability, the dashboard provides possible retention actions such as:

* Personalized offers
* Retention discounts
* Contract upgrades
* Additional customer support
* Customer engagement

---

## 🛠️ Technologies Used

```text
Python
Pandas
NumPy
Scikit-learn
Logistic Regression
SHAP
Joblib
Matplotlib
Seaborn
Streamlit
```

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── model_comparison.png
```

---

## 📓 Google Colab Notebook

The complete **Data Science and Machine Learning workflow** is available in:

```text
notebooks/customer_churn_analysis.ipynb
```

The notebook contains:

* Dataset loading
* Data cleaning
* EDA
* Visualization
* Preprocessing
* Model training
* Model comparison
* Evaluation metrics
* SHAP analysis
* Model saving

The notebook can be opened directly on GitHub to view the code and saved outputs.

---

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 📊 Key Features

| Feature             | Description                                         |
| ------------------- | --------------------------------------------------- |
| Data Analysis       | Explore customer churn patterns                     |
| Preprocessing       | Encode categorical and numerical features           |
| ML Prediction       | Predict customer churn probability                  |
| Logistic Regression | Final selected model                                |
| SHAP                | Explain individual predictions                      |
| Streamlit           | Interactive web dashboard                           |
| Risk Classification | Categorize customers based on predicted probability |
| Recommendations     | Provide possible retention actions                  |

---

## 💼 Business Use Case

A telecom company could use this system to identify customers with higher predicted churn probability and prioritize them for customer retention activities.

The system demonstrates how **Machine Learning + Explainable AI + Data Visualization + Web Deployment** can be combined into a practical business application.

---

## 👩‍💻 Author

**Sumaiya Azad**

Computer Science & Engineering

---

## ⭐ Project Highlights

```text
Machine Learning
        +
Data Science
        +
Explainable AI
        +
Streamlit
        =
End-to-End Customer Churn Prediction System
```

