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
# You might need to adjust these ranges/defaults based on your training data
experience = st.slider(
    "Years of Experience", 
    min_value=0.0, 
    max_value=30.0, 
    value=5.0, 
    step=0.5
)

age = st.number_input(
    "Age (Years)", 
    min_value=18, 
    max_value=65, 
    value=30, 
    step=1
)

education_level = st.selectbox(
    "Education Level Encoding", 
    options=[1, 2, 3, 4], # Example: 1=Bachelors, 2=Masters, 3=PhD, etc.
    index=1,
    help="Select the numerical encoding used for education level in the model training."
)


# --- 3. Prediction Logic ---
if st.button("Predict Salary"):
    # Create the feature vector (must be in the same order as model training: Experience, Education_Level, Age)
    features = np.array([[experience, education_level, age]])
    
    # Make prediction
    try:
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Format the output (adjust currency and formatting as needed)
        # Using a simple integer format for demonstration
        st.success(f"The predicted salary is: **${prediction:,.2f}**")
        st.balloons()
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Model trained using scikit-learn.")
