import streamlit as st
import pickle
import numpy as np

# Load the models
with open(r'D:\Completed Projects\Heart_disease_predicter---Ml-Practice-main\Heart_Dieases.pkl', 'rb') as f:
    model_13_features = pickle.load(f)

with open(r'D:\Completed Projects\Heart_disease_predicter---Ml-Practice-main\Heart_Dieases_6.pkl', 'rb') as f:
    model_6_features = pickle.load(f)

# Define the input fields for both models with their ranges as information
input_fields_13 = {
    'Age': '0 to 200',
    'Sex': '0 or 1',
    'Chest Pain Type': '0, 1, 2, 3',
    'Resting Blood Pressure': '80 to 230',
    'Serum Cholesterol': '100 to 650',
    'Fasting Blood Sugar': '0 or 1',
    'Resting ECG': '0, 1, 2',
    'Max Heart Rate Achieved': '60 to 220',
    'Exercise Induced Angina': '0 or 1',
    'ST Depression': '0.0 to 10.0',
    'Slope of Peak Exercise ST Segment': '0, 1, 2',
    'Number of Major Vessels': '0, 1, 2, 3, 4',
    'Thal': '0, 1, 2, 3'
}

input_fields_6 = {
    'Age': '0 to 200',
    'Resting Blood Pressure': '80 to 230',
    'Serum Cholesterol': '100 to 650',
    'Max Heart Rate Achieved': '60 to 220',
    'ST Depression': '0.0 to 10.0',
    'Thal': '0, 1, 2, 3'
}

# Custom prediction function for model with 6 features
def predict_(model, features):
    prob = model.predict_proba([features])[:, 1]
    prediction = np.where(prob > 0.4, 1, 0)
    return prediction[0]

# Streamlit app
st.title("Heart Disease Prediction")

# Select the model
model_choice = st.selectbox("Choose the model", ["Model with 13 features", "Model with 6 features"])

# Collect user inputs based on model choice
if model_choice == "Model with 13 features":
    st.subheader("Enter the following 13 features:")
    inputs = []
    for label, info in input_fields_13.items():
        st.text(f"{label} ({info})")
        inputs.append(st.number_input(f"Enter {label}", key=label))
    if st.button("Predict"):
        inputs = np.array(inputs).reshape(1, -1)  # Reshape to match the model's expected input shape
        prediction = model_13_features.predict(inputs)[0]
        if prediction == 1:
            st.error("There is a possibility of heart disease.")
        else:
            st.success("You are free from heart disease.")
else:
    st.subheader("Enter the following 6 features:")
    inputs = []
    for label, info in input_fields_6.items():
        st.text(f"{label} ({info})")
        inputs.append(st.number_input(f"Enter {label}", key=label))
    if st.button("Predict"):
        inputs = np.array(inputs).reshape(1, -1)  # Reshape to match the model's expected input shape
        prediction = predict_(model_6_features, inputs[0])
        if prediction == 1:
            st.error("There is a possibility of heart disease.")
        else:
            st.success("You are free from heart disease.")

