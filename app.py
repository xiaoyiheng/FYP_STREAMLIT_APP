import streamlit as st
import gdown
import pickle
import pandas as pd
import numpy as np
from scipy.special import boxcox


# Google Drive URL to the pickle file
url = 'https://drive.google.com/uc?id=1-UbbDBp7dpWTucdik-CNzpHRXEH3BLgJ'
output = 'my_pickle_file.pkl'

# Download the pickle file from Google Drive
@st.cache(allow_output_mutation=True)
def download_model():
    gdown.download(url, output, quiet=False)
    with open(output, 'rb') as file:
        loaded_objects = pickle.load(file)
    return loaded_objects

# Load the necessary objects (
objects = download_model()
model = objects['model']
min_max_scaler = objects['min_max_scaler']
boxcox_transformer = objects['boxcox_transformer']
yeo_johnson_transformer = objects['yeo_johnson_transformer']

# Streamlit app title
st.title('CREDIT CARD APPROVAL PREDICTION')
st.write('This app predicts if an applicant will be approved for a credit card or not. Just fill in the following information and click on the Predict button.')

# Input features
# Gender
gender = st.radio('Gender', ['Male', 'Female'])

# Age
age = st.slider('Age', 18, 70, 25) # default to 25

# Marital Status
marital_status = st.selectbox(
    'Marital_Status',
    ['Civil marriage', 'Married', 'Separated', 'Single', 'Widow']
)

# Housing Type
housing_type = st.selectbox(
    'Housing_Type',
    ['Co-op apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents', 'Apartment']
)

# Total Annual Income
total_income = st.number_input('Income (in MYR)', min_value=0, value=50000)

# Income Type
income_type = st.selectbox(
    'Income_Type',
    ['Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working']
)

# Employment Length
years_employed = st.slider('Years_Employed', 0, 40, 10)

# Education Level
education_level = st.selectbox(
    'Education_Level',
    ['Lower secondary', 'Secondary', 'Incomplete higher', 'Higher education', 'Academic degree']
)

# Car ownership
car_ownership = st.radio('Own_Car', ['Yes', 'No'])

# Real Estate ownership
realty_ownership = st.radio('Own_Realty', ['Yes', 'No'])

# Work Phone Ownership
work_phone_ownership = st.radio('Has_Work_Phone', ['Yes', 'No'])

# Phone Ownership
phone_ownership = st.radio('Has_Phone', ['Yes', 'No'])

# Email Ownership
email_ownership = st.radio('Has_Email', ['Yes', 'No'])


# Preprocessing
def preprocess_input(gender, age, marital_status, housing_type, total_income, income_type, years_employed, education_level, car_ownership, realty_ownership, work_phone_ownership, phone_ownership, email_ownership):
    # Map categorical features
    gender_mapping = {'Male': 0, 'Female': 1}
    marital_status_mapping = {'Civil marriage': 0, 'Married': 1, 'Separated': 2, 'Single': 3, 'Widow': 4}
    housing_type_mapping = {'Co-op apartment': 0, 'Municipal apartment': 1, 'Office apartment': 2, 'Rented apartment': 3, 'With parents': 4, 'Apartment': 5}
    income_type_mapping = {'Commercial associate': 0, 'Pensioner': 1, 'State servant': 2, 'Student': 3, 'Working': 4}
    education_level_mapping = {'Lower secondary': 0, 'Secondary': 1, 'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4}

    # Map the categorical variables
    gender = gender_mapping[gender]
    marital_status = marital_status_mapping[marital_status]
    housing_type = housing_type_mapping[housing_type]
    income_type = income_type_mapping[income_type]
    education_level = education_level_mapping[education_level]
    car_ownership = 1 if car_ownership == 'Yes' else 0
    realty_ownership = 1 if realty_ownership == 'Yes' else 0
    work_phone_ownership = 1 if work_phone_ownership == 'Yes' else 0
    phone_ownership = 1 if phone_ownership == 'Yes' else 0
    email_ownership = 1 if email_ownership == 'Yes' else 0

    # Apply transformations
    age_scaled = min_max_scaler.transform(np.array([[age]]))  # Min-Max scaling for age
    
    # Apply Box-Cox using the lambda value
    boxcox_lambda = objects['boxcox_transformer']  # Get the saved lambda from the pickle file
    total_income_boxcox = boxcox(total_income, boxcox_lambda)  # Apply Box-Cox directly with the lambda

    # Apply Yeo-Johnson transformation for Years_Employed
    years_employed_yeojohnson = objects['yeo_johnson_transformer'].transform(np.array([[years_employed]]))

    # Combine all features into a single array
    features = np.array([[gender, age_scaled[0][0], marital_status, housing_type, total_income_boxcox, income_type,
                          years_employed_yeojohnson[0][0], education_level, car_ownership, realty_ownership,
                          work_phone_ownership, phone_ownership, email_ownership]])

    # Convert to a DataFrame and handle missing values
    features_df = pd.DataFrame(features, columns=[
        'gender', 'age_scaled', 'marital_status', 'housing_type', 'total_income_boxcox', 'income_type',
        'years_employed_yeojohnson', 'education_level', 'car_ownership', 'realty_ownership',
        'work_phone_ownership', 'phone_ownership', 'email_ownership'
    ])
    
    # Handle missing values by filling them 
    features_df.fillna(0, inplace=True)  

    return features_df.values

# Preprocess the input
preprocessed_data = preprocess_input(gender, age, marital_status, housing_type, total_income, income_type, years_employed, education_level, car_ownership, realty_ownership, work_phone_ownership, phone_ownership, email_ownership)

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(preprocessed_data)
    probability = model.predict_proba(preprocessed_data)[:, 1]
    
    if prediction == 1:
        st.write(f"The applicant is unlikely to be approved for a credit card with a probability of {probability[0]:.2f}")
    else:
        st.write(f"The applicant is likely to be approved for a credit card with a probability of {probability[0]:.2f}")
