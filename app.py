import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model
import joblib

# Load preprocessing objects and model
preprocessor = joblib.load('preprocessor.pkl')  # Ensure preprocessor.pkl is saved correctly
model = load_model('ABHIDL.h5')  # Ensure model.h5 is saved correctly

# Define categorical and numerical features
categorical_features = ['Age', 'Race', 'Gender']
numerical_features = ['Phyiscal Health', 'Mental Health', 'Dental Health', 'Employment',
                      'Stress Keeps Patient from Sleeping', 'Medication Keeps Patient from Sleeping',
                      'Pain Keeps Patient from Sleeping', 'Bathroom Needs Keeps Patient from Sleeping',
                      'Uknown Keeps Patient from Sleeping', 'Trouble Sleeping', 'Prescription Sleep Medication']

st.title('Doctor Visits Prediction')

# Input fields for user data
age = st.number_input('Age', min_value=0, value=0)
race = st.selectbox('Race', options=['Race1', 'Race2', 'Race3'])  # Replace with actual categories
gender = st.selectbox('Gender', options=['Male', 'Female'])

phy_health = st.number_input('Phyiscal Health', min_value=0, value=0)
mental_health = st.number_input('Mental Health', min_value=0, value=0)
dental_health = st.number_input('Dental Health', min_value=0, value=0)
employment = st.selectbox('Employment', options=['Employed', 'Unemployed'])  # Replace with actual categories
stress_sleep = st.number_input('Stress Keeps Patient from Sleeping', min_value=0, value=0)
medication_sleep = st.number_input('Medication Keeps Patient from Sleeping', min_value=0, value=0)
pain_sleep = st.number_input('Pain Keeps Patient from Sleeping', min_value=0, value=0)
bathroom_sleep = st.number_input('Bathroom Needs Keeps Patient from Sleeping', min_value=0, value=0)
unknown_sleep = st.number_input('Uknown Keeps Patient from Sleeping', min_value=0, value=0)
trouble_sleep = st.number_input('Trouble Sleeping', min_value=0, value=0)
prescription_sleep = st.number_input('Prescription Sleep Medication', min_value=0, value=0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Age': [age],
    'Race': [race],
    'Gender': [gender],
    'Phyiscal Health': [phy_health],
    'Mental Health': [mental_health],
    'Dental Health': [dental_health],
    'Employment': [employment],
    'Stress Keeps Patient from Sleeping': [stress_sleep],
    'Medication Keeps Patient from Sleeping': [medication_sleep],
    'Pain Keeps Patient from Sleeping': [pain_sleep],
    'Bathroom Needs Keeps Patient from Sleeping': [bathroom_sleep],
    'Uknown Keeps Patient from Sleeping': [unknown_sleep],
    'Trouble Sleeping': [trouble_sleep],
    'Prescription Sleep Medication': [prescription_sleep]
})

# Preprocess the input data
try:
    input_data_preprocessed = preprocessor.transform(input_data)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")

# Make a prediction
try:
    prediction = model.predict(input_data_preprocessed)
    predicted_class = np.argmax(prediction)
    st.write(f'Predicted class: {predicted_class}')
except Exception as e:
    st.error(f"Error during prediction: {e}")

