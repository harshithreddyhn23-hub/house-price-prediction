import os
import subprocess
import joblib

# if model not present, train it
if not os.path.exists("house_model.pkl"):
    subprocess.run(["python", "train_model.py"])

model = joblib.load("house_model.pkl")
import streamlit as st
import joblib
import pandas as pd

# load trained model
model = joblib.load("house_model.pkl")

st.title("🏠 California House Price Prediction")

st.write("Enter house details")

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=800)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.5)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"]
)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "longitude":[longitude],
        "latitude":[latitude],
        "housing_median_age":[housing_median_age],
        "total_rooms":[total_rooms],
        "total_bedrooms":[total_bedrooms],
        "population":[population],
        "households":[households],
        "median_income":[median_income],
        "ocean_proximity":[ocean_proximity]
    })

    # convert categorical to dummy variables
    input_data = pd.get_dummies(input_data)

    # add missing columns
    required_columns = model.feature_names_in_

    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # reorder columns
    input_data = input_data[required_columns]

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")