import streamlit as st
import pickle
import numpy as np

# --- 1. Load the Model ---
# Ensure 'Student_model.pkl' is in the same directory as this app.py file
try:
    with open('Student_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'Student_model.pkl' not found. Please ensure the model file is in the same directory.")
    # Stop the app execution if the model can't be loaded
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Streamlit UI Elements ---
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💰 Simple Salary Prediction App")
st.markdown("Enter the required details to get an estimated salary prediction.")
st.markdown("---")

# Input fields for the features
st.header("Employee Details")

# Assuming Education_Level is categorical (e.g., encoded 1, 2, 3)
# Mapping for better user experience, assuming 1, 2, 3, 4 are the encoded values
EDUCATION_MAP = {
    "High School": 1,
    "Bachelors": 2,
    "Masters": 3,
    "PhD/Doctorate": 4
}
EDUCATION_OPTIONS = list(EDUCATION_MAP.keys())


# Input for Experience (first feature in your model)
experience = st.slider(
    "Years of Experience", 
    min_value=0.0, 
    max_value=30.0, 
    value=5.0, 
    step=0.5
)

# Input for Age (third feature in your model)
age = st.number_input(
    "Age (Years)", 
    min_value=18, 
    max_value=65, 
    value=30, 
    step=1
)

# Input for Education Level (second feature in your model, using the map)
selected_education = st.selectbox(
    "Education Level", 
    options=EDUCATION_OPTIONS,
    index=1, # Default to Bachelors
    help="Select the educational background."
)

# Convert the selected string back to the numerical encoding the model expects
education_level_encoded = EDUCATION_MAP[selected_education]


# --- 3. Prediction Logic ---
if st.button("Predict Salary", type="primary"):
    
    # Create the feature vector 
    # CRITICAL: The order MUST match the model's training order: Experience, Education_Level, Age
    features = np.array([[experience, education_level_encoded, age]])
    
    # Make prediction
    try:
        # Predict the salary
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Display the formatted result
        st.success(f"The estimated salary is: **${prediction:,.2f}**")
        st.balloons()
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Model trained using scikit-learn (Features: Experience, Education_Level, Age)")
