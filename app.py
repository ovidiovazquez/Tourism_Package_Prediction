
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" /
    "tourism_random_forest_pipeline.joblib"
)

SCHEMA_PATH = BASE_DIR / "models" / "model_schema.json"


# ============================================================
# LOAD MODEL AND INPUT SCHEMA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    schema = load_schema()

except Exception as error:
    st.error(
        "Unable to load the model or input schema. "
        "Please verify the deployment files."
    )
    st.exception(error)
    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("Tourism Package Prediction")

st.markdown(
    """
    Predict whether a customer is likely to purchase a
    tourism package using a trained Random Forest model.

    Enter the customer information below and click
    **Predict Purchase** to generate a prediction.
    """
)

st.info(
    "This application provides decision-support predictions "
    "based on customer characteristics and sales interaction data."
)


# ============================================================
# CUSTOMER INPUT FORM
# ============================================================

categorical_options = schema["categorical_options"]

with st.form("customer_prediction_form"):

    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        gender = st.selectbox(
            "Gender",
            categorical_options["Gender"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            categorical_options["MaritalStatus"]
        )

        occupation = st.selectbox(
            "Occupation",
            categorical_options["Occupation"]
        )

        designation = st.selectbox(
            "Designation",
            categorical_options["Designation"]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

    with col2:

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=2
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=10,
            value=0
        )

        number_of_trips = st.number_input(
            "Number of Trips per Year",
            min_value=0,
            max_value=50,
            value=3
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star Rating",
            [3, 4, 5]
        )

        passport = st.selectbox(
            "Has a Passport?",
            [0, 1],
            format_func=lambda value:
                "Yes" if value == 1 else "No"
        )

        own_car = st.selectbox(
            "Owns a Car?",
            [0, 1],
            format_func=lambda value:
                "Yes" if value == 1 else "No"
        )

    with col3:

        type_of_contact = st.selectbox(
            "Type of Contact",
            categorical_options["TypeofContact"]
        )

        product_pitched = st.selectbox(
            "Product Pitched",
            categorical_options["ProductPitched"]
        )

        duration_of_pitch = st.number_input(
            "Duration of Sales Pitch (minutes)",
            min_value=0.0,
            max_value=180.0,
            value=15.0
        )

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=20,
            value=3
        )

        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score",
            [1, 2, 3, 4, 5]
        )

    submitted = st.form_submit_button(
        "Predict Purchase",
        use_container_width=True
    )


# ============================================================
# PREPARE CUSTOMER INPUT DATAFRAME
# ============================================================

if submitted:

    customer_data = {
        "Age": float(age),
        "TypeofContact": type_of_contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(duration_of_pitch),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(
            number_of_person_visiting
        ),
        "NumberOfFollowups": float(
            number_of_followups
        ),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(
            preferred_property_star
        ),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(number_of_trips),
        "Passport": int(passport),
        "PitchSatisfactionScore": int(
            pitch_satisfaction_score
        ),
        "OwnCar": int(own_car),
        "NumberOfChildrenVisiting": float(
            number_of_children_visiting
        ),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income)
    }

    # Build input DataFrame using the training feature order
    input_df = pd.DataFrame(
        [customer_data],
        columns=schema["features"]
    )

    # Verify feature compatibility
    if list(input_df.columns) != schema["features"]:
        st.error("Input features do not match the model schema.")
        st.stop()

    # ========================================================
    # GENERATE PREDICTION
    # ========================================================

    try:

        prediction = int(
            model.predict(input_df)[0]
        )

        purchase_probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        st.divider()

        st.subheader("Prediction Results")

        if prediction == 1:

            st.success(
                "Prediction: Customer is likely to purchase "
                "the tourism package."
            )

        else:

            st.warning(
                "Prediction: Customer is unlikely to purchase "
                "the tourism package."
            )

        st.metric(
            "Estimated Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )

        st.progress(purchase_probability)

        with st.expander("View Customer Input Data"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(error)


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "Advanced Machine Learning and MLOps | "
    "Tourism Package Prediction | Random Forest Classifier"
)
