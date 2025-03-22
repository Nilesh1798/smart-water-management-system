import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def clean_feature_names(df):
    df.columns = [re.sub(r'[\[\]<> ,]', '_', col) for col in df.columns]
    return df

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Generate complete timestamps from 11/10/2014 to 02/04/2020 (every 8 hours)
date_range = pd.date_range(start="2014-10-11 00:00:00", end="2020-04-02 00:00:00", freq='8H')
expected_timestamps = pd.DataFrame({'Timestamp': date_range.strftime('%d/%m/%Y %H')})

# Convert Humidity from percentage to numeric
train_data['Humidity'] = pd.to_numeric(train_data['Humidity'].astype(str).str.rstrip('%'), errors='coerce')
test_data['Humidity'] = pd.to_numeric(test_data['Humidity'].astype(str).str.rstrip('%'), errors='coerce')

# Handle categorical data with one-hot encoding
categorical_cols = ['Apartment_Type', 'Income_Level', 'Amenities']
train_data = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Ensure train and test have the same feature columns
train_data = clean_feature_names(train_data)
test_data = clean_feature_names(test_data)
train_cols = train_data.columns.drop('Water_Consumption')
test_data = test_data.reindex(columns=train_cols, fill_value=0)

# Drop missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Split data
X = train_data.drop(columns=['Water_Consumption', 'Timestamp'])
y = train_data['Water_Consumption']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 5,
    'tree_method': 'hist'
}
model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, "validation")], early_stopping_rounds=10)

# Validate model
y_pred = model.predict(dval)
score = max(0, 100 - np.sqrt(mean_squared_error(y_val, y_pred)))
print(f'Validation Score: {score}')

# Predict on test data
dtest = xgb.DMatrix(test_data.drop(columns=['Timestamp'], errors='ignore'))
test_predictions = model.predict(dtest)

# Create submission DataFrame
submission = pd.DataFrame({
    'Timestamp': test_data['Timestamp'],
    'Water_Consumption': test_predictions
})

# ✅ Ensure all timestamps exist in the submission
submission = expected_timestamps.merge(submission, on="Timestamp", how="left")

# Fill missing values with the mean prediction
submission['Water_Consumption'].fillna(submission['Water_Consumption'].mean(), inplace=True)

# Ensure exactly 6000 rows
submission = submission.head(6000)

# Save submission file
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved successfully!")
print(f"Final Submission file shape: {submission.shape}")
print(submission.head())
