import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD TRAINED LOGISTIC REGRESSION MODEL
# ============================================================

model = joblib.load(
    "models/churn_model.pkl"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Customer Churn Prediction Dashboard"
)

st.write(
    "Predict the probability that a customer will churn "
    "based on their service and account information."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "👤 Customer Information"
)


# ------------------------------------------------------------
# Demographic Information
# ------------------------------------------------------------

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)


senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)


# ------------------------------------------------------------
# Service Information
# ------------------------------------------------------------

tenure = st.sidebar.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)


phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


online_backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device_protection = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tech_support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


# ------------------------------------------------------------
# Account Information
# ------------------------------------------------------------

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)


total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


# ============================================================
# CREATE CUSTOMER DATAFRAME
# ============================================================

customer = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless],

    "PaymentMethod": [payment],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]
})


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Customer Churn",
    type="primary"
):

    # Get churn probability
    probability = model.predict_proba(
        customer
    )[0][1]

    # Save probability so it remains available
    # when Streamlit reruns the application
    st.session_state["probability"] = probability


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

if "probability" in st.session_state:

    probability = (
        st.session_state["probability"]
    )

    probability_percentage = (
        probability * 100
    )


    # --------------------------------------------------------
    # Risk Classification
    # --------------------------------------------------------

    if probability >= 0.5:

        risk = "High Risk"

    else:

        risk = "Low Risk"


    # --------------------------------------------------------
    # Result Header
    # --------------------------------------------------------

    st.subheader(
        "🎯 Prediction Result"
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk
        )


    # --------------------------------------------------------
    # Result Message
    # --------------------------------------------------------

    if probability >= 0.5:

        st.error(
            "⚠️ This customer is likely to churn."
        )

    else:

        st.success(
            "✅ This customer is unlikely to churn."
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.divider()

if st.button(
    "📈 Explain Prediction"
):

    # --------------------------------------------------------
    # Check whether prediction exists
    # --------------------------------------------------------

    if "probability" not in st.session_state:

        st.warning(
            "⚠️ Please click 'Predict Customer Churn' "
            "first."
        )

        st.stop()


    st.subheader(
        "📈 Why did the model make this prediction?"
    )


    # ========================================================
    # GET MODEL COMPONENTS
    # ========================================================

    preprocessor = model.named_steps[
        "preprocessor"
    ]


    # Logistic Regression classifier
    logistic_model = model.named_steps[
        "classifier"
    ]


    # ========================================================
    # LOAD ORIGINAL DATASET
    # ========================================================

    data = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )


    # ========================================================
    # PREPARE BACKGROUND DATA
    # ========================================================

    # Remove target column
    X_background = data.drop(
        columns=["Churn"],
        errors="ignore"
    )


    # Remove customer ID if it exists
    X_background = X_background.drop(
        columns=["customerID"],
        errors="ignore"
    )


    # ========================================================
    # SAMPLE BACKGROUND CUSTOMERS
    # ========================================================

    X_background = X_background.sample(
        n=min(200, len(X_background)),
        random_state=42
    )


    # ========================================================
    # TRANSFORM BACKGROUND DATA
    # ========================================================

    background_transformed = (
        preprocessor.transform(
            X_background
        )
    )


    # ========================================================
    # TRANSFORM CURRENT CUSTOMER
    # ========================================================

    customer_transformed = (
        preprocessor.transform(
            customer
        )
    )


    # ========================================================
    # GET FEATURE NAMES
    # ========================================================

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # ========================================================
    # SHAP LINEAR EXPLAINER
    # ========================================================

    explainer = shap.LinearExplainer(
        logistic_model,
        background_transformed
    )


    # ========================================================
    # CALCULATE SHAP VALUES
    # ========================================================

    shap_values = explainer(
        customer_transformed
    )


    # ========================================================
    # GET CURRENT CUSTOMER SHAP VALUES
    # ========================================================

    shap_values_single = (
        shap_values.values[0]
    )


    # ========================================================
    # CREATE IMPORTANCE DATAFRAME
    # ========================================================

    importance = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": shap_values_single

    })


    # Absolute value is useful for ranking
    importance["Absolute SHAP"] = (
        importance["SHAP Value"].abs()
    )


    # ========================================================
    # SELECT TOP 10 FEATURES
    # ========================================================

    importance = importance.sort_values(
        "Absolute SHAP",
        ascending=False
    ).head(10)


    # ========================================================
    # CLEAN FEATURE NAMES
    # ========================================================

    importance["Feature"] = (
        importance["Feature"]
        .str.replace(
            "cat__",
            "",
            regex=False
        )
        .str.replace(
            "num__",
            "",
            regex=False
        )
    )


    # ========================================================
    # SORT FOR VISUALIZATION
    # ========================================================

    importance = importance.sort_values(
        "SHAP Value"
    )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.write(
        "### 🔎 Top Factors Affecting This Prediction"
    )


    display_table = importance[
        [
            "Feature",
            "SHAP Value"
        ]
    ].copy()


    display_table["SHAP Value"] = (
        display_table["SHAP Value"]
        .round(4)
    )


    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SHAP GRAPH
    # ========================================================

    st.write(
        "### 📊 Feature Contribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.barh(
        importance["Feature"],
        importance["SHAP Value"]
    )


    # Zero reference line
    ax.axvline(
        0,
        linewidth=1
    )


    ax.set_xlabel(
        "SHAP Value"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Top Factors Influencing Customer Churn"
    )


    plt.tight_layout()


    st.pyplot(fig)


    # ========================================================
    # SHAP INTERPRETATION
    # ========================================================

    st.info(
        "Positive SHAP values push the prediction "
        "toward higher churn probability, while "
        "negative SHAP values push it toward lower "
        "churn probability. Larger absolute SHAP "
        "values indicate stronger influence."
    )


    # ========================================================
    # MOST IMPORTANT POSITIVE FACTOR
    # ========================================================

    positive_features = importance[
        importance["SHAP Value"] > 0
    ]


    negative_features = importance[
        importance["SHAP Value"] < 0
    ]


    if len(positive_features) > 0:

        strongest_positive = (
            positive_features
            .sort_values(
                "SHAP Value",
                ascending=False
            )
            .iloc[0]
        )

        st.write(
            f"🔴 **Strongest factor increasing churn:** "
            f"{strongest_positive['Feature']} "
            f"(SHAP = "
            f"{strongest_positive['SHAP Value']:.4f})"
        )


    if len(negative_features) > 0:

        strongest_negative = (
            negative_features
            .sort_values(
                "SHAP Value"
            )
            .iloc[0]
        )

        st.write(
            f"🟢 **Strongest factor reducing churn:** "
            f"{strongest_negative['Feature']} "
            f"(SHAP = "
            f"{strongest_negative['SHAP Value']:.4f})"
        )


# ============================================================
# BUSINESS RECOMMENDATION
# ============================================================

if "probability" in st.session_state:

    probability = (
        st.session_state["probability"]
    )


    st.divider()

    st.subheader(
        "💡 Recommended Action"
    )


    if probability >= 0.7:

        st.warning(
            "🚨 High churn risk. Consider offering "
            "a retention discount, contract upgrade, "
            "personalized offer, or additional "
            "customer support."
        )


    elif probability >= 0.4:

        st.info(
            "⚠️ Moderate churn risk. Monitor this "
            "customer and consider improving "
            "engagement or providing a personalized offer."
        )


    else:

        st.success(
            "✅ Low churn risk. Continue regular "
            "customer engagement and service."
        )