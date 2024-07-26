import pandas as pd
import numpy as np
import streamlit as st
from catboost import CatBoostClassifier
import joblib
import os
import matplotlib.pyplot as plt
import shap

# Paths
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)
MODEL_PATH = "C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\model\\catboost_model.cbm"
DATA_PATH = "C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(page_title="Churn Project")

@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        st.stop()
    return data

@st.cache_resource
def load_x_y(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    model.load_model(MODEL_PATH)
    return model

def preprocess_data(df):
    # Ensure all categorical features are strings
    cat_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Convert categorical columns to strings
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Handle NaN values in categorical columns
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Convert all other columns to strings if they are not categorical
    for col in df.columns:
        if col not in cat_features:
            df[col] = df[col].astype(str)
    
    return df

def calculate_shap(model, X_train, X_test):
    # Preprocess data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_test, customer_id, X_test):
    # Visualize SHAP values for a specific customer
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_index], X_test.iloc[customer_index], link="logit")
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar")
    summary_fig = plt.gcf()
    st.pyplot(summary_fig)
    plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    
    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, customer_id, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    
    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_test, customer_id, X_test)

    # Waterfall
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[customer_index], feature_names=X_test.columns, max_display=20)

st.title("Telco Customer Churn Project")

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\X_train.pkl")
    X_test = load_x_y("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\X_test.pkl")
    y_train = load_x_y("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\y_train.pkl")
    y_test = load_x_y("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\y_test.pkl")

    customer_ids = load_x_y("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\customer_ids.pkl") if os.path.exists("C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\customer_ids.pkl") else []
    
    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()

    # Radio buttons for options
    election = st.radio("Make Your Choice:", ("Feature Importance", "User-based SHAP", "Calculate the probability of CHURN"))
    
    # If User-based SHAP option is selected
    if election == "User-based SHAP":
        # Customer ID text input
        customer_id = st.selectbox("Choose the Customer", customer_ids)
        customer_index = customer_ids.index(customer_id) if customer_id in customer_ids else None
        
        if customer_index is not None:
            st.write(f'Customer {customer_id}: Actual value for the Customer Churn : {y_test.iloc[customer_index]}')
            y_pred = model.predict(X_test)
            st.write(f"Customer {customer_id}: CatBoost Model's prediction for the Customer Churn : {y_pred[customer_index]}")
            plot_shap(model, data, customer_id, X_train=X_train, X_test=X_test)
    
    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

    # If Calculate CHURN Probability option is selected
    elif election == "Calculate the probability of CHURN":
        # Retrieving data from the user
        customerID = st.text_input("Customer ID", "6464-UIAEA")
        gender = st.selectbox("Gender:", ("Female", "Male"))
        senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
        partner = st.selectbox("Partner:", ("No", "Yes"))
        dependents = st.selectbox("Dependents:", ("No", "Yes"))
        tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure, step=1)
        phone_service = st.selectbox("PhoneService:", ("No", "Yes"))
        multiple_lines = st.selectbox("MultipleLines:", ("No", "Yes"))
        internet_service = st.selectbox("InternetService:", ("No", "DSL", "Fiber optic"))
        online_security = st.selectbox("OnlineSecurity:", ("No", "Yes"))
        online_backup = st.selectbox("OnlineBackup:", ("No", "Yes"))
        device_protection = st.selectbox("DeviceProtection:", ("No", "Yes"))
        tech_support = st.selectbox("TechSupport:", ("No", "Yes"))
        streaming_tv = st.selectbox("StreamingTV:", ("No", "Yes"))
        streaming_movies = st.selectbox("StreamingMovies:", ("No", "Yes"))
        contract = st.selectbox("Contract:", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("PaperlessBilling", ("No", "Yes"))
        payment_method = st.selectbox("PaymentMethod:", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
        monthly_charges = st.number_input("MonthlyCharges:", min_value=0.0, max_value=max_monthly_charges, step=0.01)
        total_charges = st.number_input("TotalCharges:", min_value=0.0, max_value=max_total_charges, step=0.01)

        if st.button("Predict"):
            new_customer_data = pd.DataFrame({
                "customerID": [customerID],
                "gender_Female": [1 if gender == "Female" else 0],
                "gender_Male": [1 if gender == "Male" else 0],
                "SeniorCitizen": [senior_citizen],
                "Partner_No": [1 if partner == "No" else 0],
                "Partner_Yes": [1 if partner == "Yes" else 0],
                "Dependents_No": [1 if dependents == "No" else 0],
                "Dependents_Yes": [1 if dependents == "Yes" else 0],
                "Tenure": [tenure],
                "PhoneService_No": [1 if phone_service == "No" else 0],
                "PhoneService_Yes": [1 if phone_service == "Yes" else 0],
                "MultipleLines_No": [1 if multiple_lines == "No" else 0],
                "MultipleLines_Yes": [1 if multiple_lines == "Yes" else 0],
                "InternetService_No": [1 if internet_service == "No" else 0],
                "InternetService_DSL": [1 if internet_service == "DSL" else 0],
                "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
                "OnlineSecurity_No": [1 if online_security == "No" else 0],
                "OnlineSecurity_Yes": [1 if online_security == "Yes" else 0],
                "OnlineBackup_No": [1 if online_backup == "No" else 0],
                "OnlineBackup_Yes": [1 if online_backup == "Yes" else 0],
                "DeviceProtection_No": [1 if device_protection == "No" else 0],
                "DeviceProtection_Yes": [1 if device_protection == "Yes" else 0],
                "TechSupport_No": [1 if tech_support == "No" else 0],
                "TechSupport_Yes": [1 if tech_support == "Yes" else 0],
                "StreamingTV_No": [1 if streaming_tv == "No" else 0],
                "StreamingTV_Yes": [1 if streaming_tv == "Yes" else 0],
                "StreamingMovies_No": [1 if streaming_movies == "No" else 0],
                "StreamingMovies_Yes": [1 if streaming_movies == "Yes" else 0],
                "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
                "Contract_One year": [1 if contract == "One year" else 0],
                "Contract_Two year": [1 if contract == "Two year" else 0],
                "PaperlessBilling_No": [1 if paperless_billing == "No" else 0],
                "PaperlessBilling_Yes": [1 if paperless_billing == "Yes" else 0],
                "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
                "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],
                "PaymentMethod_Bank transfer (automatic)": [1 if payment_method == "Bank transfer (automatic)" else 0],
                "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges]
                
            })
            # Ensure customerID is string
            new_customer_data['customerID'] = new_customer_data['customerID'].astype(str)
            # Convert max_total_charges to float if it's not already
            max_total_charges = float(max_total_charges)

            # Use the converted value in the number_input function
            total_charges = st.number_input("TotalCharges:", min_value=0.0, max_value=max_total_charges, step=0.01)
            def preprocess_data(df):
            # Convert 'True'/'False' strings to boolean values
                for col in df.select_dtypes(include=['object']):
                    if df[col].str.contains('True|False').any():
                        df[col] = df[col].replace({'True': 1, 'False': 0})
                    else:
                        # Convert other columns to numeric where possible
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                return df

            
            # Preprocess new customer data
            new_customer_data = preprocess_data(new_customer_data)
            
            # Make a prediction
            prediction = model.predict_proba(new_customer_data)[0, 1]  # Probability of churn
            st.write(f"Probability of Churn for Customer {customerID}: {prediction:.2%}")

if __name__ == "__main__":
    main()
