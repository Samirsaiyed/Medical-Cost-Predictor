import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to numerical using Label Encoding
    label_encoder = LabelEncoder()
    data['sex'] = label_encoder.fit_transform(data['sex'])
    data['smoker'] = label_encoder.fit_transform(data['smoker'])
    data['region'] = label_encoder.fit_transform(data['region'])
    return data

# Function to predict medical cost
def predict_medical_cost(model, input_data):
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
def main():
    st.title("Medical Cost Prediction App")

    # Input form
    st.sidebar.header("User Input Features")
    
    age = st.sidebar.slider("Age", 18, 64, 25)
    sex = st.sidebar.radio("Sex", ["male", "female"])
    bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
    children = st.sidebar.slider("Number of Children", 0, 5, 1)
    smoker = st.sidebar.radio("Smoker", ["yes", "no"])
    region = st.sidebar.radio("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Create a dictionary with user inputs
    user_input = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}

    # Create a DataFrame with a single row (user input)
    input_df = pd.DataFrame([user_input])

    # Make prediction
    prediction = predict_medical_cost(model, input_df)

    # Display prediction
    st.subheader("Predicted Medical Cost:")
    st.write(f"${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
