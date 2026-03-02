import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb 
st.title("Welcome to Energy Predictor Application")
st.write("Lets predict the appicance energy consumption based on the temperature of the device.")
model = jb.load(r"appliance_energy_predictor.pkl")
#take the input
temp = st.number_input("Enter the temperature of the device: ",min_value=0.0, max_value=46.0, value=5.0)
#create button to predict energy consumption
if st.button("Predict Energy Consumption"):
    new_data = np.array([[temp]])
    prediction = model.predict(new_data)
    st.write(f"The predicted energy consumption is: {prediction[0]:.2f} kWh")
st.write("Thanks for using the Energy Predictor Application!")