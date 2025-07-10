import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# File path constants
TRAIN_CSV_PATH = "/kaggle/input/toxic_cmt_classify/jigsaw-toxic-comment-train.csv"
TEST_CSV_PATH = "/kaggle/input/toxic_cmt_classify/test.csv"
PROCESSED_DIR = "./processed"

# Auto-detect file paths if defaults are not found
def find_file(file_name):
    for root, dirs, files in os.walk("/kaggle/input/toxic_cmt_classify"):
        for file in files:
            if file_name.lower() in file.lower():
                return os.path.join(root, file)
    return None

TRAIN_CSV_PATH = find_file("train") or TRAIN_CSV_PATH
TEST_CSV_PATH = find_file("test") or TEST_CSV_PATH

# Load datasets
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

# Identify target column
target_column = 'toxic'

# Preprocessing function
def preprocess_data(train_df, test_df):
    # Identify feature columns
    feature_columns = train_df.columns.difference([target_column])

    # Define numerical, categorical, and text columns
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = []  # No categorical columns in the train set
    text_cols = ['comment_text']

    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('text', text_transformer, text_cols)
        ],
        remainder='drop'  # Drop other columns
    )

    # Fit and transform the training data
    X_train = preprocessor.fit_transform(train_df[feature_columns])
    y_train = train_df[target_column]

    # Transform the test data
    X_test = preprocessor.transform(test_df[feature_columns])

    # Create processed DataFrames
    train_df_processed = pd.DataFrame(X_train, columns=numerical_cols + ['tfidf_' + str(i) for i in range(X_train.shape[1] - len(numerical_cols))])
    train_df_processed[target_column] = y_train.values

    test_df_processed = pd.DataFrame(X_test, columns=numerical_cols + ['tfidf_' + str(i) for i in range(X_test.shape[1] - len(numerical_cols))])

    return train_df_processed, test_df_processed

# Process the data
train_df_processed, test_df_processed = preprocess_data(train_df, test_df)

# Save processed data
os.makedirs(PROCESSED_DIR, exist_ok=True)
train_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'train_processed.csv'), index=False)
test_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'test_processed.csv'), index=False)

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
RANDOM_STATE = 42
MODEL_PATH = "./models/model.pkl"

# Check if 'toxic' column exists
if 'toxic' not in train_df_processed.columns:
    raise KeyError("The target column 'toxic' is not found in the processed DataFrame.")

# Separate features and target
X = train_df_processed['comment_text']
y = train_df_processed['toxic']

# Stratified split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)

# Build a model (using RandomForest as an example)
model = RandomForestClassifier(random_state=RANDOM_STATE)

# Fit the model on the training split
model.fit(X_train_vectorized, y_train)

# Predict on validation set
y_val_pred = model.predict_proba(X_val_vectorized)[:, 1]

# Evaluate and print metrics
accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))
f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
logloss = log_loss(y_val, y_val_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Log Loss: {logloss}")

# Persist the trained model
joblib.dump(model, MODEL_PATH)

# Return the trained model instance
trained_model = model

import pandas as pd
import joblib
import os

# Constants
TEST_PROCESSED_PATH = "./processed/test_processed.csv"
SUBMISSION_PATH = "./outputs/submission.csv"

# Load the trained model
if 'trained_model' not in locals() or trained_model is None:
    trained_model = joblib.load("./models/model.pkl")

# Load the processed test data
test_df_processed = pd.read_csv(TEST_PROCESSED_PATH)

# Generate predictions
predictions = trained_model.predict_proba(test_df_processed['comment_text'].values.reshape(-1, 1))[:, 1]

# Build submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df_processed.index,
    'toxic': predictions
})

# Save the submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file saved to: {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
