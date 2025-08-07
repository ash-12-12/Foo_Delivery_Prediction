import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- 0. IMPORTANT: Add the CapperTransformer class definition here ---
# This class must be defined in the same script that loads the model.
class CapperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_factor=1.5):
        self.iqr_factor = iqr_factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        else:
            self.columns = range(X.shape[1])
            
        for i, col in enumerate(self.columns):
            Q1 = np.quantile(X[col] if isinstance(X, pd.DataFrame) else X[:, i], 0.25)
            Q3 = np.quantile(X[col] if isinstance(X, pd.DataFrame) else X[:, i], 0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.iqr_factor * IQR
            self.upper_bounds_[col] = Q3 + self.iqr_factor * IQR
        return self

    def transform(self, X, y=None):
        X_capped = X.copy()
        
        for i, col in enumerate(self.columns):
            if isinstance(X_capped, pd.DataFrame):
                X_capped[col] = np.where(
                    X_capped[col] < self.lower_bounds_[col],
                    self.lower_bounds_[col],
                    X_capped[col]
                )
                X_capped[col] = np.where(
                    X_capped[col] > self.upper_bounds_[col],
                    self.upper_bounds_[col],
                    X_capped[col]
                )
            else:
                col_data = X_capped[:, i]
                X_capped[:, i] = np.where(
                    col_data < self.lower_bounds_[col],
                    self.lower_bounds_[col],
                    col_data
                )
                X_capped[:, i] = np.where(
                    X_capped[:, i] > self.upper_bounds_[col],
                    self.upper_bounds_[col],
                    X_capped[:, i]
                )
        return X_capped


# --- 1. Load the trained model pipeline ---
try:
    loaded_pipeline = joblib.load('final_model.joblib')
except FileNotFoundError:
    st.error("Error: 'final_model.joblib' not found. Please ensure it's in the same directory.")
    st.stop()

# --- 2. Set up the Streamlit app interface ---
st.set_page_config(page_title="Food Delivery Time Predictor", layout="centered")
st.title("🍔 Food Delivery Time Predictor")
st.markdown("Enter the details of a new order to predict its delivery time.")

# --- 3. Create input widgets for each feature ---
st.subheader("Order Details")
distance = st.slider("Distance (km)", min_value=1.0, max_value=50.0, value=15.0, step=0.1)
prep_time = st.slider("Preparation Time (min)", min_value=5, max_value=40, value=20)
courier_exp = st.slider("Courier Experience (years)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

st.subheader("Environmental Conditions")
weather = st.selectbox("Weather", options=['Sunny', 'Rainy', 'Cloudy'])
traffic_level = st.selectbox("Traffic Level", options=['Low', 'Medium', 'High'])
time_of_day = st.selectbox("Time of Day", options=['Morning', 'Afternoon', 'Evening'])
vehicle_type = st.selectbox("Vehicle Type", options=['Bike', 'Car', 'Truck'])


# --- 4. Create a prediction button ---
if st.button("Predict Delivery Time"):
    # Create a DataFrame from the user inputs
    new_data = pd.DataFrame({
        'Distance_km': [distance],
        'Weather': [weather],
        'Traffic_Level': [traffic_level],
        'Time_of_Day': [time_of_day],
        'Vehicle_Type': [vehicle_type],
        'Preparation_Time_min': [prep_time],
        'Courier_Experience_yrs': [courier_exp],
    })

    try:
        log_prediction = loaded_pipeline.predict(new_data)
        predicted_delivery_time = np.exp(log_prediction)[0]
        
        st.success(f"Predicted Delivery Time: **{predicted_delivery_time:.2f} minutes**")
        st.info("This prediction is based on a trained Linear Regression model.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")