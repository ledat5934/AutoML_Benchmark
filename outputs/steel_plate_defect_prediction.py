import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# File-path constants
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "test.csv"
PROCESSED_DIR = "./processed"

# Auto-detect file paths if not found
def find_csv_files(base_path):
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                yield os.path.join(dirpath, filename)

try:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
except FileNotFoundError:
    csv_files = list(find_csv_files("/kaggle/input/steel_plate_defect_prediction"))
    train_df = pd.read_csv(next(f for f in csv_files if 'train' in f.lower()))
    test_df = pd.read_csv(next(f for f in csv_files if 'test' in f.lower()))

# Identify target column
target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
X_train = train_df.drop(columns=target_columns)
y_train = train_df[target_columns]

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data
X_test = test_df.drop(columns=target_columns, errors='ignore')  # Ensure test data has the same structure
X_test_processed = preprocessor.transform(X_test)

# Create processed DataFrames
train_df_processed = pd.DataFrame(X_train_processed, columns=numerical_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)))
train_df_processed[target_columns] = y_train.reset_index(drop=True)

test_df_processed = pd.DataFrame(X_test_processed, columns=numerical_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)))

# Save processed DataFrames
os.makedirs(PROCESSED_DIR, exist_ok=True)
train_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'train_processed.csv'), index=False)
test_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'test_processed.csv'), index=False)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from lightgbm import LGBMClassifier
import joblib

# Constants
MODEL_PATH = "./models/model.pkl"
RANDOM_STATE = 42

# Separate features and target
X = train_df_processed.drop(columns=target_columns)
y = train_df_processed[target_columns]

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y.values.argmax(axis=1))

# Build model
model = LGBMClassifier()

# Fit model
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

# Predictions
y_val_pred = model.predict_proba(X_val)

# Evaluate metrics
accuracy = accuracy_score(y_val.values.argmax(axis=1), y_val_pred.argmax(axis=1))
f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int), average='macro')
logloss = log_loss(y_val, y_val_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Log Loss: {logloss}')

# Save model
joblib.dump(model, MODEL_PATH)

# Return trained model
trained_model = model

import pandas as pd
import joblib

# Constants
TEST_PROCESSED_PATH = "./processed/test_processed.csv"
SUBMISSION_PATH = "./outputs/submission.csv"

# Load the test data
test_df_processed = pd.read_csv(TEST_PROCESSED_PATH)

# Load the trained model
trained_model = joblib.load("./models/model.pkl")

# Ensure the preprocessor is also saved and loaded
preprocessor = joblib.load("./models/preprocessor.pkl")

# Transform the test data
X_test_processed = preprocessor.transform(test_df_processed)

# Generate predictions
predictions = trained_model.predict_proba(X_test_processed)

# Create submission DataFrame
submission_df = pd.DataFrame(predictions, columns=['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])
submission_df.insert(0, 'id', test_df_processed.index)

# Save submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f'Submission file saved to: {SUBMISSION_PATH}')

if __name__ == "__main__":
    main()
