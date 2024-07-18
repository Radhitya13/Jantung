import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the cleaned dataset
df = pd.read_csv("honda_car_selling.csv")

# Load your trained model (assuming lr is your trained LinearRegression model)
# You should include the model training code here if you want to deploy with a pre-trained model.

# Function to predict price
def predict_price(year, kms_driven, fuel_type, suspension, car_model):
    # Perform any necessary preprocessing here (like converting categorical variables to dummy variables)
    # Assuming you have done preprocessing before model deployment

    # Example of using the model for prediction
    input_data = np.array([[year, kms_driven, fuel_type, suspension, car_model]])
    prediction = predict(input_data)
    return prediction

# Streamlit UI
def main():
    st.title("Car Price Predictor")
    st.sidebar.title("Input Parameters")

    # Define inputs using Streamlit components
    year = st.sidebar.slider("Year", min_value=1990, max_value=2023, step=1)
    kms_driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=200000, step=1000)
    fuel_type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
    suspension = st.sidebar.selectbox("Suspension", df["Suspension"].unique())
    car_model = st.sidebar.selectbox("Car Model", df["Car Model"].unique())

    # When the user clicks the predict button
    if st.sidebar.button("Predict"):
        prediction = predict_price(year, kms_driven, fuel_type, suspension, car_model)
        st.write(f"Predicted Price: {prediction}")

if __name__ == "__main__":
    main()
