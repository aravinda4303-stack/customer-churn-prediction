from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# SHAP is optional. Predictions still work if SHAP is unavailable.
try:
    import shap
    SHAP_READY = True
    SHAP_ERROR = None
except Exception as exc:
    shap = None
    SHAP_READY = False
    SHAP_ERROR = str(exc)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "customer_churn_xgboost_reduced16.json"
PREPROCESSOR_PATH = BASE_DIR / "customer_churn_preprocessor_reduced16.pkl"


# These must match the 16 features used when training the model.
FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Monthly Charges"
]


# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_PATH.name}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # Check raw input features.
    saved_features = list(
        getattr(preprocessor, "feature_names_in_", [])
    )

    if saved_features and saved_features != FEATURES:
        raise ValueError(
            "The preprocessor does not match this 16-feature application."
            f"\nExpected: {FEATURES}"
            f"\nFound: {saved_features}"
        )

    # Check transformed feature count.
    processed_count = len(
        preprocessor.get_feature_names_out()
    )

    model_feature_count = model.get_booster().num_features()

    if processed_count != model_feature_count:
        raise ValueError(
            f"The preprocessor creates {processed_count} features, "
            f"but the model expects {model_feature_count}."
        )

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()

except Exception as exc:
    st.error("The prediction system could not be loaded.")
    st.code(str(exc))

    st.info(
        "Make sure these two files are in the same folder as app.py:\n"
        "customer_churn_xgboost_reduced16.json\n"
        "customer_churn_preprocessor_reduced16.pkl"
    )

    st.stop()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def yes_no(label, key, help_text=None):
    """Return 1 for Yes and 0 for No."""

    choice = st.selectbox(
        label,
        options=["No", "Yes"],
        key=key,
        help=help_text
    )

    return 1 if choice == "Yes" else 0


def build_customer_dataframe(customer_values):
    """Create the input row in the correct training feature order."""

    return pd.DataFrame(
        [[customer_values[feature] for feature in FEATURES]],
        columns=FEATURES
    )


def explain_prediction(processed_data, customer_values):
    """Generate SHAP contributions grouped by original input feature."""

    if not SHAP_READY:
        return None, (
            "SHAP is unavailable. Install it with: "
            "py -m pip install shap"
        )

    try:
        # Convert the transformed row into a dense array for SHAP.
        if hasattr(processed_data, "toarray"):
            dense_data = processed_data.toarray()
        else:
            dense_data = np.asarray(processed_data)

        dense_data = np.asarray(dense_data)

        if dense_data.ndim == 1:
            dense_data = dense_data.reshape(1, -1)

        transformed_names = list(
            preprocessor.get_feature_names_out()
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dense_data)

        # Some SHAP versions return a list for classification.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)
        feature_count = len(transformed_names)

        # Handle SHAP outputs with an explicit class dimension.
        if shap_values.ndim == 3:

            if shap_values.shape[1] == feature_count:
                shap_values = shap_values[0, :, -1]

            elif shap_values.shape[2] == feature_count:
                shap_values = shap_values[0, -1, :]

            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {shap_values.shape}"
                )

        elif shap_values.ndim == 2:
            shap_values = shap_values[0]

        elif shap_values.ndim != 1:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_values.shape}"
            )

        if len(shap_values) != feature_count:
            raise ValueError(
                "The number of SHAP values does not match "
                "the transformed feature count."
            )

        # Combine one-hot encoded columns under their original feature.
        grouped_values = {
            feature: 0.0 for feature in FEATURES
        }

        def identify_original_feature(transformed_name):

            clean_name = str(transformed_name).split("__", 1)[-1]

            for feature in sorted(FEATURES, key=len, reverse=True):

                if clean_name == feature:
                    return feature

                if clean_name.startswith(feature + "_"):
                    return feature

            return None

        for transformed_name, contribution in zip(
            transformed_names,
            shap_values
        ):

            original_feature = identify_original_feature(
                transformed_name
            )

            if original_feature is not None:
                grouped_values[original_feature] += float(contribution)

        rows = []

        for feature, contribution in grouped_values.items():

            if contribution > 0:
                direction = "Pushes toward churn"

            elif contribution < 0:
                direction = "Pushes away from churn"

            else:
                direction = "Little or no impact"

            rows.append({
                "Feature": feature,
                "Customer Value": customer_values[feature],
                "SHAP Contribution": contribution,
                "Effect": direction,
                "Absolute Impact": abs(contribution)
            })

        explanation = pd.DataFrame(rows)

        explanation = explanation.sort_values(
            "Absolute Impact",
            ascending=False
        ).reset_index(drop=True)

        return explanation, None

    except Exception as exc:
        return None, (
            "The prediction succeeded, but SHAP could not create "
            f"an explanation: {exc}"
        )


# ============================================================
# 4. WEBSITE HEADER
# ============================================================

st.title("📊 Customer Churn Intelligence")

st.write(
    "Predict whether a telecom customer is likely to leave "
    "and understand which factors influenced the prediction."
)

st.caption(
    "Machine Learning: XGBoost | Input Features: 16 | "
    "Explainability: SHAP"
)

st.info(
    "This application uses the reduced 16-feature model. "
    "The dataset's charge values are in US dollars, and its "
    "Yes/No service fields are represented as 1/0."
)


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About This Project")

    st.write(
        "Customer Churn Intelligence estimates the probability "
        "that a telecom customer will leave the service."
    )

    st.subheader("Model Information")

    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**Input features:** 16")
    st.write("**Target:** Customer Churn")
    st.write("**Explainability:** SHAP")

    st.divider()

    st.subheader("Previously Recorded Test Metrics")

    st.metric("Accuracy", "75.02%")
    st.metric("Precision", "51.90%")
    st.metric("Recall", "80.21%")
    st.metric("F1-score", "63.03%")
    st.metric("ROC-AUC", "85.30%")

    st.caption(
        "These are previously recorded test results, not a "
        "guarantee of performance on new customers."
    )


# ============================================================
# 6. CUSTOMER PROFILE
# ============================================================

st.header("1. Customer Profile")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        key="gender_input"
    )

    senior_citizen = yes_no(
        "Is the customer a senior citizen?",
        "senior_input"
    )

    partner = yes_no(
        "Does the customer have a partner?",
        "partner_input"
    )

with col2:

    dependents = yes_no(
        "Does the customer have dependents?",
        "dependents_input"
    )

    tenure_months = st.number_input(
        "Time with Company (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1,
        key="tenure_input",
        help="For example, 2 years with the company equals 24 months."
    )


st.divider()


# ============================================================
# 7. PHONE AND INTERNET SERVICES
# ============================================================

st.header("2. Customer Services")

col1, col2 = st.columns(2)

with col1:

    phone_service = yes_no(
        "Does the customer have phone service?",
        "phone_input"
    )

    if phone_service == 1:

        multiple_lines = yes_no(
            "Does the customer have multiple phone lines?",
            "multiple_lines_input"
        )

    else:

        multiple_lines = 0

        st.caption(
            "Multiple phone lines is set to No because "
            "phone service is disabled."
        )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"],
        key="internet_input"
    )


if internet_service == "No":

    online_security = 0
    online_backup = 0
    device_protection = 0
    tech_support = 0
    streaming_tv = 0
    streaming_movies = 0

    st.info(
        "Internet add-on services are automatically set to No "
        "because the customer has no internet service."
    )

else:

    st.subheader("Internet Add-on Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = yes_no(
            "Online Security",
            "security_input"
        )

        online_backup = yes_no(
            "Online Backup",
            "backup_input"
        )

    with col2:

        device_protection = yes_no(
            "Device Protection",
            "device_input"
        )

        tech_support = yes_no(
            "Technical Support",
            "tech_support_input"
        )

    with col3:

        streaming_tv = yes_no(
            "Streaming TV",
            "streaming_tv_input"
        )

        streaming_movies = yes_no(
            "Streaming Movies",
            "streaming_movies_input"
        )


st.divider()


# ============================================================
# 8. CONTRACT AND BILLING
# ============================================================

st.header("3. Contract and Billing")

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract Type",
        options=[
            "Month-to-month",
            "One year",
            "Two year"
        ],
        key="contract_input"
    )


with col2:

    billing_frequency = st.selectbox(
        "Billing Amount Frequency",
        options=["Monthly", "Yearly"],
        key="billing_frequency_input",
        help=(
            "The model was trained using Monthly Charges. "
            "Yearly amounts will be divided by 12."
        )
    )


if billing_frequency == "Monthly":

    entered_charges = st.number_input(
        "Monthly Charges (USD)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        format="%.2f",
        key="monthly_charges_input"
    )

    monthly_charges = float(entered_charges)


else:

    entered_charges = st.number_input(
        "Yearly Charges (USD)",
        min_value=0.0,
        max_value=2400.0,
        value=840.0,
        step=12.0,
        format="%.2f",
        key="yearly_charges_input",
        help="For example, $840 per year is converted to $70 per month."
    )

    monthly_charges = float(entered_charges) / 12


st.caption(
    f"Monthly amount used by the model: "
    f"**${monthly_charges:,.2f} per month**"
)


# ============================================================
# 9. PREPARE THE CUSTOMER INPUT
# ============================================================

customer_values = {

    "Gender": gender,

    "Senior Citizen": int(senior_citizen),

    "Partner": int(partner),

    "Dependents": int(dependents),

    "Tenure Months": int(tenure_months),

    "Phone Service": int(phone_service),

    "Multiple Lines": int(multiple_lines),

    "Internet Service": internet_service,

    "Online Security": int(online_security),

    "Online Backup": int(online_backup),

    "Device Protection": int(device_protection),

    "Tech Support": int(tech_support),

    "Streaming TV": int(streaming_tv),

    "Streaming Movies": int(streaming_movies),

    "Contract": contract,

    "Monthly Charges": monthly_charges
}


# Detect whether the user has changed inputs after prediction.
current_signature = tuple(
    (feature, str(customer_values[feature]))
    for feature in FEATURES
)


# ============================================================
# 10. PREDICT CUSTOMER CHURN
# ============================================================

st.divider()

predict_clicked = st.button(
    "🔍 Predict Customer Churn",
    type="primary",
    use_container_width=True,
    key="predict_button"
)

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


# No form is used here. This avoids the undefined 'submitted' error.
if predict_clicked:

    try:

        customer_df = build_customer_dataframe(customer_values)

        # Apply the same preprocessing pipeline used during training.
        processed_data = preprocessor.transform(customer_df)

        # Predict class and probability.
        prediction = int(
            model.predict(processed_data)[0]
        )

        probabilities = model.predict_proba(processed_data)[0]

        churn_probability = float(probabilities[1])

        # Calculate an explanation. If this fails, the prediction remains usable.
        explanation, explanation_error = explain_prediction(
            processed_data,
            customer_values
        )

        st.session_state["last_prediction"] = {

            "signature": current_signature,

            "prediction": prediction,

            "probability": churn_probability,

            "customer_values": customer_values.copy(),

            "explanation": explanation,

            "explanation_error": explanation_error
        }

    except Exception as exc:

        st.error("Prediction failed.")

        st.code(str(exc))


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

result = st.session_state.get("last_prediction")


if result is not None:

    if result["signature"] != current_signature:

        st.warning(
            "You changed one or more customer details. "
            "Click Predict Customer Churn again to update the results."
        )

    else:

        st.divider()

        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            if result["prediction"] == 1:

                st.error(
                    "⚠️ Prediction: Customer Likely to Churn"
                )

            else:

                st.success(
                    "✅ Prediction: Customer Likely to Stay"
                )

        with col2:

            st.metric(
                "Estimated Churn Probability",
                f"{result['probability']:.1%}"
            )


        st.write("Churn Probability")

        st.progress(
            min(1.0, max(0.0, result["probability"]))
        )

        st.caption(
            "This is a model estimate, not a guarantee of future behavior."
        )


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        st.divider()

        st.header("🧠 Why Did the Model Make This Prediction?")

        explanation = result["explanation"]

        if explanation is not None and not explanation.empty:

            st.write(
                "SHAP helps explain how the model's input features "
                "influenced this individual prediction."
            )

            st.caption(
                "Positive contributions push the model output toward churn; "
                "negative contributions push it away. The contribution values "
                "are not percentage-point changes in churn probability."
            )

            # Show the most influential features.
            top_features = explanation.head(10).copy()

            chart_data = top_features.set_index("Feature")[
                ["SHAP Contribution"]
            ]

            st.subheader("Most Influential Features")

            st.bar_chart(
                chart_data,
                height=350
            )

            # Display a detailed table.
            st.subheader("Explanation Details")

            display_table = explanation[
                [
                    "Feature",
                    "Customer Value",
                    "SHAP Contribution",
                    "Effect"
                ]
            ].copy()

            display_table["SHAP Contribution"] = (
                display_table["SHAP Contribution"].round(4)
            )

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "These explanations describe the model's behavior. "
                "They do not prove that a feature caused the customer to churn."
            )

        else:

            st.warning(
                result["explanation_error"]
                or "The SHAP explanation could not be generated."
            )

            if not SHAP_READY:
                st.code("py -m pip install shap")


        # ----------------------------------------------------
        # CUSTOMER INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander("View Customer Information Used"):

            customer_summary = pd.DataFrame([
                {
                    "Feature": feature,
                    "Value": result["customer_values"][feature]
                }
                for feature in FEATURES
            ])

            st.dataframe(
                customer_summary,
                use_container_width=True,
                hide_index=True
            )