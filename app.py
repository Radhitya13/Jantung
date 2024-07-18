import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the cleaned dataset
df = pd.read_csv("honda_car_selling.csv")

# Train the LinearRegression model
X = df[["Year", "kms_Driven", "Fuel_Type", "Suspension", "Car_Model"]]
y = df["Selling_Price"]
lr = LinearRegression()
lr.fit(X, y)

# Function to predict price
def predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model):
    # Perform any necessary preprocessing here (like converting categorical variables to dummy variables)
    # Assuming you have done preprocessing before model deployment

    # Example of using the model for prediction
    input_data = np.array([[Year, kms_Driven, Fuel_Type, Suspension, Car_Model]])
    prediction = lr.predict(input_data)
    return prediction

# Streamlit UI
def main():
    st.title("Car Price Predictor")
    st.sidebar.title("Input Parameters")

    # Define inputs using Streamlit components
    Year = st.sidebar.slider("Year", min_value=1990, max_value=2023, step=1)
    kms_Driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=200000, step=1000)
    Fuel_Type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
    Suspension = st.sidebar.selectbox("Suspension", df["Suspension"].unique())
    Car_Model = st.sidebar.selectbox("Car Model", df["Car Model"].unique())

    # When the user clicks the predict button
    if st.sidebar.button("Predict"):
        prediction = predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model)
        st.write(f"Predicted Price: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
