"""
import streamlit as st
import pandas as pd
import pickle  # or pickle if you used that for saving the model

# Load your trained model
# Replace 'your_model.pkl' with your actual model file
model = pickle.load('career.pkl')

# Function to preprocess input data
def preprocess_input(input_df):
    # Add your actual preprocessing steps here (e.g., one-hot encoding)
    # This should match the preprocessing you did during training
    
    # Example preprocessing:
    categorical_cols = ['Gender', 'Field_of_Study', 'University_Location', 
                       'Career_Interests', 'Employment_Type', 
                       'Entrepreneurial_Aspirations']
    
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Ensure all columns from training are present
    # (You might need to align columns with your training data)
    return input_encoded

# Streamlit app
def main():
    st.set_page_config(page_title="Career Path Predictor", page_icon="🚀")
    
    st.title("Career Path Prediction App")
    st.write("Fill in the details to get career path recommendations")
    
    with st.form("career_form"):
        # Personal Information
        age = st.number_input("Age", min_value=18, max_value=50, value=22)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Education Information
        field_of_study = st.selectbox("Field of Study", 
                                     ["Science", "Business", "Engineering", "Law", "Arts"])
        university_location = st.selectbox("University Location", ["A", "B", "C"])
        gpa = st.slider("GPA", 1.0, 4.0, 3.0)
        relevant_coursework = st.selectbox("Relevant Coursework Completed?", [1, 0])
        
        # Experience Information
        prior_employment = st.selectbox("Prior Employment Experience?", [1, 0])
        entrepreneurial_exp = st.selectbox("Entrepreneurial Experience?", [1, 0])
        startup_participation = st.selectbox("Startup Participation?", [1, 0])
        
        # Career Preferences
        career_interests = st.selectbox("Career Interests", 
                                      ["Design", "Tech", "Finance", "Business", "Healthcare"])
        employment_type = st.selectbox("Preferred Employment Type", 
                                     ["Full-time", "Part-time", "Internship"])
        entrepreneurial_asp = st.selectbox("Entrepreneurial Aspirations", 
                                         ["High", "Medium", "Low"])
        
        submitted = st.form_submit_button("Predict Career Path")
        
    if submitted:
        # Create input dataframe
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Field_of_Study': [field_of_study],
            'University_Location': [university_location],
            'GPA': [gpa],
            'Relevant_Coursework': [relevant_coursework],
            'Prior_Employment': [prior_employment],
            'Entrepreneurial_Experience': [entrepreneurial_exp],
            'Startup_Participation': [startup_participation],
            'Career_Interests': [career_interests],
            'Employment_Type': [employment_type],
            'Entrepreneurial_Aspirations': [entrepreneurial_asp]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Preprocess input
        processed_input = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Display result
        st.subheader("Recommended Career Path")
        st.success(f"{prediction[0]}")
        
        # Optional: Add interpretation/explanation
        st.write("Based on your profile, skills, and preferences, our AI model recommends:")

if __name__ == "__main__":
    main()

"""
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('career_pipeline.pkl')

# Input fields for user data
st.title("Career Guidance Prediction App")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=24)
gpa = st.number_input("GPA (1 to 4)", min_value=1.0, max_value=4.0, value=3.0, step=0.01)

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
field_of_study = st.selectbox("Field of Study", ["Science", "Business", "Engineering", "Law", "Arts"])
university_location = st.selectbox("University Location", ["A", "B", "C"])
relevant_coursework = st.selectbox("Relevant Coursework (Yes=1, No=0)", [0, 1])
prior_employment = st.selectbox("Prior Employment (Yes=1, No=0)", [0, 1])

# Ordinal Encoding for Employment_Type
employment_order = ['Internship', 'Part-time', 'Full-time']
employment_type = st.selectbox("Employment Type", employment_order)

entrepreneurial_experience = st.selectbox("Entrepreneurial Experience (Yes=1, No=0)", [0, 1])
startup_participation = st.selectbox("Startup Participation (Yes=1, No=0)", [0, 1])

# Career Interests
career_interests = st.selectbox("Career Interests", ["Design", "Tech", "Finance", "Business", "Healthcare"])

# Ordinal Encoding for Entrepreneurial_Aspirations
entrep_aspirations_order = ['Low', 'Medium', 'High']
entrepreneurial_aspirations = st.selectbox("Entrepreneurial Aspirations", entrep_aspirations_order)

# Convert categorical inputs into a DataFrame
user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Field_of_Study": [field_of_study],
    "University_Location": [university_location],
    "GPA": [gpa],
    "Relevant_Coursework": [relevant_coursework],
    "Prior_Employment": [prior_employment],
    "Employment_Type": [employment_type],
    "Entrepreneurial_Experience": [entrepreneurial_experience],
    "Startup_Participation": [startup_participation],
    "Career_Interests": [career_interests],
    "Entrepreneurial_Aspirations": [entrepreneurial_aspirations]
})

# Make predictions using the pipeline
if st.button("Predict Recommended Career Path"):
    prediction = pipeline.predict(user_data)[0]
    st.success(f"Recommended Career Path: **{prediction}**")
