import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
# Title for the browser tab
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# --- Load the Model ---
# Ensure 'sal prediction.pkl' is in the same directory as this app.py file
try:
    with open('sal prediction.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'sal prediction.pkl' not found. Please ensure the model file is in the same directory.")
    model = None # Set model to None to prevent errors later

# --- Helper Functions ---
def predict_salary(experience, education_level_map, age):
    """
    Makes a salary prediction using the loaded model.
    """
    if model is None:
        return None

    # Map the selected education level text to the numerical value the model expects
    education_level_numeric = education_level_map

    # The model expects a 2D array (or DataFrame) with the features in the correct order:
    # [Experience, Education_Level, Age]
    features = np.array([[experience, education_level_numeric, age]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Ensure prediction is not negative (though linear models can sometimes predict negatives)
    return max(0, prediction)

# --- Streamlit App Layout ---

## 💰 Salary Prediction Model
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>💰 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6A5ACD;'>Predict the estimated salary based on key professional metrics.</h3>", unsafe_allow_html=True)

st.write("---")

if model is not None:
    # Define the mapping for Education Level (ADJUST THIS TO MATCH YOUR MODEL'S TRAINING)
    # This dictionary maps the display name to the numerical value the model expects
    EDUCATION_LEVELS = {
        "High School": 0,
        "Bachelor's Degree": 1,
        "Master's Degree": 2,
        "Ph.D.": 3
    }
    
    # --- Input Fields ---
    with st.container(border=True):
        st.subheader("Employee Details")
        
        # 1. Experience
        experience = st.slider(
            "Years of Experience", 
            min_value=0.0, 
            max_value=30.0, 
            value=5.0, 
            step=0.5,
            help="Enter the total years of professional experience."
        )

        # 2. Age
        age = st.number_input(
            "Age", 
            min_value=18, 
            max_value=100, 
            value=25, 
            step=1,
            help="Enter the employee's current age."
        )
        
        # 3. Education Level
        selected_education_name = st.selectbox(
            "Education Level",
            options=list(EDUCATION_LEVELS.keys()),
            index=1, # Default to Bachelor's Degree
            help="Select the highest completed education level."
        )
        # Get the numerical value for the model
        education_level_value = EDUCATION_LEVELS[selected_education_name]

    st.write("") # Add some vertical space

    # --- Prediction Button and Output ---
    if st.button("Predict Salary", type="primary", use_container_width=True):
        
        # Make the prediction
        predicted_salary = predict_salary(experience, education_level_value, age)

        if predicted_salary is not None:
            
            # Format the output
            formatted_salary = f"${predicted_salary:,.2f}"

            st.success(f"### 🎉 Estimated Salary Prediction")
            st.markdown(f"""
                <div style='text-align: center; background-color: #E6F7FF; padding: 20px; border-radius: 10px; border: 2px solid #1E90FF;'>
                    <p style='font-size: 1.2em; color: #555;'>The predicted annual salary for this profile is:</p>
                    <p style='font-size: 2.5em; font-weight: bold; color: #008000;'>{formatted_salary}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info("Note: This is an estimation based on the trained Linear Regression model.")

st.write("---")
st.caption("Developed using Streamlit and Scikit-learn.")
