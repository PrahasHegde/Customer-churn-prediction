################################################## Import Libraries ##################################################
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool

################################################## Data Loading and Editing ##################################################
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Convert TotalCharges to numeric, filling NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(int)

# Convert SeniorCitizen to integer
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# Replace 'No phone service' and 'No internet service' with 'No' for certain columns
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in columns_to_replace:
    df[column] = df[column].replace('No internet service', 'No')

# Convert 'Churn' categorical variable to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

################################################## StratifiedShuffleSplit ##################################################
# Create the StratifiedShuffleSplit object
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
train_index, test_index = next(strat_split.split(df, df["Churn"]))

# Create train and test sets
strat_train_set = df.loc[train_index]
strat_test_set = df.loc[test_index]

X_train = strat_train_set.drop("Churn", axis=1)
y_train = strat_train_set["Churn"].copy()

X_test = strat_test_set.drop("Churn", axis=1)
y_test = strat_test_set["Churn"].copy()

# Convert byte strings to strings in DataFrame
def convert_bytes_to_str(df):
    for col in df.select_dtypes(include=['object']):
        if df[col].apply(lambda x: isinstance(x, bytes)).any():
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    return df

# Apply conversion
X_train = convert_bytes_to_str(X_train)
X_test = convert_bytes_to_str(X_test)

def preprocess_data(df):
    # Convert boolean-like columns to numeric
    bool_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.contains('True|False', na=False).any()]
    for col in bool_columns:
        df[col] = df[col].replace({'True': 1, 'False': 0})

    # Convert categorical columns to integers or strings
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].nunique() <= 10:  # if the number of unique values is small, use Ordinal Encoding
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(df[[col]])
        else:
            df[col] = df[col].astype(str)  # Convert to string if too many unique values

    return df


# Apply preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

################################################## CATBOOST ##################################################
# Identify categorical columns for CatBoost
categorical_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Initialize and fit CatBoostClassifier
cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

# Predict on test set
y_pred = cat_model.predict(X_test)

# Calculate evaluation metrics
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

# Create a DataFrame to store results
model_names = ['CatBoost_Model']
result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_names)

# Print results
print(result)

# Save the model in the 'model' directory
model_dir = "C:\\Users\\hegde\\OneDrive\\Desktop\\Churn Prediction\\model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "catboost_model.cbm")
cat_model.save_model(model_path)
