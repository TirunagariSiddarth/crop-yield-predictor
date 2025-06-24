import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model, scaler, and polynomial transformer
model = joblib.load('crop_yield_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

# Streamlit app
st.title("Crop Yield Predictor")
st.write("Enter the parameters below to predict crop yield (tons/ha):")

# Input fields with realistic ranges
pesticide = st.slider("Pesticide Consumption (kg/ha)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
land_size = st.slider("Land Size (ha)", min_value=0.5, max_value=2.0, value=1.2, step=0.1)
irrigation = st.slider("Irrigation (mm)", min_value=300, max_value=800, value=500, step=10)
ph_level = st.slider("Soil pH Level", min_value=5.5, max_value=7.5, value=6.5, step=0.1)

# Predict button
if st.button("Predict Crop Yield"):
    # Prepare input data
    input_data = np.array([[pesticide, land_size, irrigation, ph_level]])
    # Scale and transform inputs
    input_data_scaled = scaler.transform(input_data)
    input_data_poly = poly.transform(input_data_scaled)
    # Make prediction
    prediction = model.predict(input_data_poly)[0]
    # Display result
    st.success(f"Predicted Crop Yield: {prediction:.2f} tons/ha")