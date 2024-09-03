import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model
import joblib

# Load preprocessing objects and model
preprocessor = joblib.load('preprocessor.pkl')
model = load_model('ABHIDL.h5')

# Page title and description
st.title('Doctor Visits Prediction')
st.markdown("""
This application predicts the number of doctor visits based on various health and lifestyle factors. 
Please enter the relevant information below and click on 'Predict Doctor Visits' to see the prediction.
""")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, value=0)
    race = st.selectbox('Race', options=['Race1', 'Race2', 'Race3'])
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    phy_health = st.number_input('Phyiscal Health', min_value=0, value=0)
    mental_health = st.number_input('Mental Health', min_value=0, value=0)
    dental_health = st.number_input('Dental Health', min_value=0, value=0)
    employment = st.number_input('Employment (0=Unemployed, 1=Employed)', min_value=0, max_value=1, value=0)

with col2:
    stress_sleep = st.number_input('Stress Keeps Patient from Sleeping', min_value=0, value=0)
    medication_sleep = st.number_input('Medication Keeps Patient from Sleeping', min_value=0, value=0)
    pain_sleep = st.number_input('Pain Keeps Patient from Sleeping', min_value=0, value=0)
    bathroom_sleep = st.number_input('Bathroom Needs Keeps Patient from Sleeping', min_value=0, value=0)
    unknown_sleep = st.number_input('Uknown Keeps Patient from Sleeping', min_value=0, value=0)
    trouble_sleep = st.number_input('Trouble Sleeping', min_value=0, value=0)
    prescription_sleep = st.number_input('Prescription Sleep Medication', min_value=0, value=0)

# Prediction button
st.markdown("---")  # Horizontal line for separation
if st.button('Predict Doctor Visits'):
    try:
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'Race': [race],
            'Gender': [gender],
            'Employment': [employment],
            'Phyiscal Health': [phy_health],
            'Mental Health': [mental_health],
            'Dental Health': [dental_health],
            'Stress Keeps Patient from Sleeping': [stress_sleep],
            'Medication Keeps Patient from Sleeping': [medication_sleep],
            'Pain Keeps Patient from Sleeping': [pain_sleep],
            'Bathroom Needs Keeps Patient from Sleeping': [bathroom_sleep],
            'Uknown Keeps Patient from Sleeping': [unknown_sleep],
            'Trouble Sleeping': [trouble_sleep],
            'Prescription Sleep Medication': [prescription_sleep]
        })

        # Preprocess the input data
        input_data_preprocessed = preprocessor.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_preprocessed)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Display the result
        st.success(f'Predicted class: {predicted_class}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("""
---
*Disclaimer: This is a predictive tool and the results are for informational purposes only.*
""")
