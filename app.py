import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("honda_car_selling.csv")

X = df[["Year", "kms_Driven", "Fuel_Type", "Suspension", "Car_Model"]]
y = df["Selling_Price"]
lr = LinearRegression()
lr.fit(X, y)

def predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model):




input_data = np.array([[Year, kms_Driven, Fuel_Type, Suspension, Car_Model]])
prediction = lr.predict(input_data)
return prediction

def main():
st.title("Car Price Predictor")
st.sidebar.title("Input Parameters")

Year = st.sidebar.slider("Year", min_value=1990, max_value=2023, step=1)
kms_Driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=200000, step=1000)
Fuel_Type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
Suspension = st.sidebar.selectbox("Suspension", df["Suspension"].unique())
Car_Model = st.sidebar.selectbox("Car Model", df["Car Model"].unique())


if st.sidebar.button("Predict"):
    prediction = predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model)
    st.write(f"Predicted Price: {prediction}")
if __name__ == "__main__":
    main()
