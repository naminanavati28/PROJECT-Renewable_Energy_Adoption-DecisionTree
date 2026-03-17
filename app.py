import streamlit as st
import pandas as pd
import joblib
from src.preprocess import prepare_input

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "Renewable_Energy_Adoption_model.pkl"
    model = joblib.load(model_path)
    return model

model = load_model()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(
    page_title="Renewable Energy Adoption Prediction",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Renewable Energy Adoption Predictor")

st.write("Enter the values below to predict the adoption level.")

# ---------------------------------------------------
# User Input Fields (MUST MATCH TRAINING FEATURES)
# ---------------------------------------------------
carbon_emissions = st.number_input(
    "Carbon Emissions",
    min_value=0.0,
    max_value=10000.0,
    value=0.0
)

energy_output = st.number_input(
    "Energy Output",
    min_value=0.0,
    max_value=10000.0,
    value=0.0
)

renewability_index = st.number_input(
    "Renewability Index",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)

cost_efficiency = st.number_input(
    "Cost Efficiency",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)

# ---------------------------------------------------
# Prepare Input & Predict
# ---------------------------------------------------
if st.button("Predict Adoption Level"):
    input_df = prepare_input(
        carbon_emissions=carbon_emissions,
        energy_output=energy_output,
        renewability_index=renewability_index,
        cost_efficiency=cost_efficiency
    )

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Adoption Level: **{prediction}**")
