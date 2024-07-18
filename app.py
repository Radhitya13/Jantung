import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the cleaned dataset
df = pd.read_csv("honda_car_selling.csv")

# Load your trained model (assuming lr is your trained LinearRegression model)
# You should include the model training code here if you want to deploy with a pre-trained model.

X = df[["Year", "kms Driven", "Fuel Type", "Suspension", "Car Model"]]
y = df["Price"] 

categorical_cols = ["Fuel Type", "Suspension", "Car Model"]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),
    ], remainder='passthrough')

X = preprocessor.fit_transform(X)

lr = LinearRegression()
lr.fit(X, y)

# Function to predict price
def predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model):
    # Perform any necessary preprocessing here (like converting categorical variables to dummy variables)
    # Assuming you have done preprocessing before model deployment

    # Example of using the model for prediction
    input_data = np.array([[Year, kms_Driven, Fuel_Type, Suspension, Car_Model]])
    input_data_transformed = preprocessor.transform(input_data)
    prediction = lr.predict(input_data_transformed)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Car Price Predictor")
    st.write("Masukkan detail mobil Anda untuk memprediksi harga.")

    # Define inputs using Streamlit components
    Year = st.sidebar.slider("Year", min_value=1990, max_value=2023, step=1)
    kms_Driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=200000, step=1000)
    Fuel_Type = st.sidebar.selectbox("Fuel Type", df["Fuel Type"].unique())
    Suspension = st.sidebar.selectbox("Suspension", df["Suspension"].unique())
    Car_Model = st.sidebar.selectbox("Car Model", df["Car Model"].unique())

    # When the user clicks the predict button
    if st.sidebar.button("Predict"):
        prediction = predict_price(Year, kms_Driven, Fuel_Type, Suspension, Car_Model)
        st.write(f"Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    main()
