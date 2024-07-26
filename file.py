import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Path to your dataset
DATA_PATH = "C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Path where you want to save the .pkl files
DATA_SAVE_PATH = "C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\data\\"

# Ensure the directory exists
if not os.path.exists(DATA_SAVE_PATH):
    os.makedirs(DATA_SAVE_PATH)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Print out the first few rows and column names to inspect
print("First few rows of the dataset:")
print(data.head())

print("\nColumn names in the dataset:")
print(data.columns)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check if 'Churn' column exists in the dataframe
if 'Churn' in data.columns:
    # Drop customerID column if it exists
    if 'customerID' in data.columns:
        customer_ids = data['customerID']
        data.drop('customerID', axis=1, inplace=True)
    else:
        customer_ids = None

    # Convert categorical variables to numerical
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.fillna(0, inplace=True)
    data = pd.get_dummies(data)

    # Check the dataframe after preprocessing
    print("\nData after preprocessing:")
    print(data.head())
    print("\nColumn names after preprocessing:")
    print(data.columns)

    # Use one-hot encoded Churn column
    if 'Churn_Yes' in data.columns:
        X = data.drop(['Churn_Yes', 'Churn_No'], axis=1)
        y = data['Churn_Yes']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save the split data to .pkl files
        joblib.dump(X_train, os.path.join(DATA_SAVE_PATH, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(DATA_SAVE_PATH, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(DATA_SAVE_PATH, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(DATA_SAVE_PATH, "y_test.pkl"))

        if customer_ids is not None:
            joblib.dump(customer_ids, os.path.join(DATA_SAVE_PATH, "customer_ids.pkl"))

        print("Data saved successfully!")
    else:
        print("One-hot encoded columns for 'Churn' not found in the dataset.")
else:
    print("Column 'Churn' not found in the dataset.")
