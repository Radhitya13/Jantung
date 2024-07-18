import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Ganti path file sesuai dengan lokasi dataset Anda
df = pd.read_csv("honda_car_selling.csv")

# Pastikan nama kolom yang dipilih sesuai dengan dataset
X = df[["Year", "kms Driven", "Fuel Type", "Suspension", "Car Model"]]
y = df["Price"]  # Ganti dengan nama kolom yang sesuai di dataset

lr = LinearRegression()
lr.fit(X, y)

def predict_price(Year, kms Driven, Fuel Type, Suspension, Car Model):
    input_data = np.array([[Year, kms Driven, Fuel Type, Suspension, Car Model]])
    prediction = lr.predict(input_data)
    return prediction[0]

def main():
    st.title("Car Price Predictor")
    st.write("Masukkan detail mobil Anda untuk memprediksi harga.")

    Year = st.sidebar.slider("Year", min_value=1990, max_value=2023, step=1)
    kms_driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=200000, step=1000)
    fuel_type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
    suspension = st.sidebar.selectbox("Suspension", df["Suspension"].unique())
    car_model = st.sidebar.selectbox("Car Model", df["Car Model"].unique())

    if st.sidebar.button("Predict"):
        prediction = predict_price(Year, kms Driven, Fuel Type, Suspension, Car Model)
        st.write(f"Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    main()
