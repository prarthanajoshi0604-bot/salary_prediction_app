import streamlit as st
import pickle
import numpy as np

# --- Configuration and Styling ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for a clean and attractive look
st.markdown("""
    <style>
    .main-header {
        font-size: 3em !important;
        font-weight: 700;
        color: #2E86C1; /* A nice blue color */
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.5em !important;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 10px;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #1A5276;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
    }
    .prediction-box {
        background-color: #D6EAF8; /* Light blue background */
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pickled model file."""
    try:
        with open('Student_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'Student_model.pkl' not found. Please ensure it is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}. This might be due to an incompatible scikit-learn version.")
        st.info("Try ensuring your installed scikit-learn version matches the one used to create the model.")
        return None

model = load_model()

# --- Application UI and Logic ---

if model is not None:
    
    st.markdown('<p class="main-header">🎓 Student Performance Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Estimate the final score based on student profile features.</p>', unsafe_allow_html=True)

    st.write("---")

    # Input Section
    st.header("👤 Enter Student Profile Details")

    # Use columns for a neat, multi-column layout for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        # Assuming Experience is in years (can be fractional)
        experience = st.number_input(
            "Experience (Years in Program/Field)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.5,
            format="%.1f",
            help="The student's professional or academic experience."
        )

    with col2:
        # Education Level: Assuming an ordinal scale (e.g., 1=High School, 2=Associate, 3=Bachelor's, 4=Master's, 5=PhD)
        education_level = st.selectbox(
            "Education Level (Ordinal Scale)",
            options=[1, 2, 3, 4, 5],
            index=2, # Default to 3 (Bachelor's or equivalent)
            format_func=lambda x: f"Level {x}",
            help="Higher number represents a higher level of education (e.g., 3=Bachelor's, 5=PhD)."
        )

    with col3:
        # Age
        age = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=100,
            value=25,
            step=1,
            help="The student's current age."
        )

    st.write("")
    st.write("---")

    # Prediction Button and Logic
    if st.button("Calculate Predicted Score"):
        
        # Prepare the input data for the model
        # The feature names in the model are: Experience, Education_Level, Age
        features = np.array([[experience, education_level, age]])

        try:
            # Make prediction
            prediction = model.predict(features)[0]

            # Display results in an attractive box
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>🧠 Prediction Result</h2>
                    <p style='font-size: 20px; color: #154360;'>
                        Based on the profile entered, the estimated performance score is:
                    </p>
                    <div style='font-size: 4em; font-weight: 900; color: #2C3E50;'>
                        {prediction:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            
            # Optional: Add context to the prediction
            st.info(f"Note: This score is a prediction from a Linear Regression model, which estimates a continuous outcome variable (like a final grade or performance index).")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check the console for detailed error information.")

    st.markdown("<footer><p style='text-align: center; color: #AAB7B8; margin-top: 50px;'>Powered by Streamlit & Scikit-learn</p></footer>", unsafe_allow_html=True)
