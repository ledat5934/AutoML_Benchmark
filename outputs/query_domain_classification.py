import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# File-path constants
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "test.csv"
PROCESSED_DIR = "./processed"

# Fallback for file paths
def find_csv_files(base_path, keyword):
    for root, _, files in os.walk(base_path):
        for file in files:
            if keyword.lower() in file.lower() and (file.endswith('.csv') or file.endswith('.parquet')):
                return os.path.join(root, file)
    return None

TRAIN_CSV_PATH = find_csv_files("/kaggle/input/query_domain_classification", "train") or TRAIN_CSV_PATH
TEST_CSV_PATH = find_csv_files("/kaggle/input/query_domain_classification", "test") or TEST_CSV_PATH

# Constants
RANDOM_STATE = 42

def load_and_preprocess_data():
    # Load datasets
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Identify target column
    target_column = 'Domain'  # Based on the EDA report
    feature_columns = train_df.columns[train_df.columns != target_column].tolist()

    # Identify numerical, categorical, and text columns
    numerical_cols = train_df.select_dtypes(include=['int64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    text_cols = ['Title']  # Based on the EDA report

    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first'))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('text', text_transformer, text_cols)
        ]
    )

    # Fit and transform the training data
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform the test data
    X_test = test_df[feature_columns]
    X_test_processed = preprocessor.transform(X_test)

    # Create processed DataFrames
    train_df_processed = pd.DataFrame(X_train_processed)
    test_df_processed = pd.DataFrame(X_test_processed)

    # Include the target column in the train DataFrame
    train_df_processed[target_column] = y_train.values

    # Save processed DataFrames
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'train_processed.csv'), index=False)
    test_df_processed.to_csv(os.path.join(PROCESSED_DIR, 'test_processed.csv'), index=False)

    return train_df_processed, test_df_processed

def main():
    load_and_preprocess_data()

if __name__ == "__main__":
    main()

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
MODEL_PATH = "./models/model.pkl"

# Prepare data
X = train_df_processed['Title']
y = train_df_processed['Domain']

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class QueryDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {**encoding, 'labels': torch.tensor(label)}

# Encode labels
label_to_id = {label: idx for idx, label in enumerate(y.unique())}
y_train_encoded = y_train.map(label_to_id).values
y_val_encoded = y_val.map(label_to_id).values

# Create datasets
train_dataset = QueryDataset(X_train.tolist(), y_train_encoded)
val_dataset = QueryDataset(X_val.tolist(), y_val_encoded)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fit the model
trainer.train()

# Evaluate
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)

# Metrics
accuracy = accuracy_score(y_val_encoded, preds)
f1 = f1_score(y_val_encoded, preds, average='weighted')
logloss = log_loss(y_val_encoded, predictions.predictions)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Log Loss: {logloss}")

# Save model
joblib.dump(model, MODEL_PATH)

trained_model = model

import joblib
import pandas as pd

# Constants
TEST_PROCESSED_PATH = "./processed/test_processed.csv"
SUBMISSION_PATH = "./outputs/submission.csv"

# Load the model if not already loaded
if trained_model is None:
    trained_model = joblib.load("models/model.pkl")

# Load the processed test data
test_df_processed = pd.read_csv(TEST_PROCESSED_PATH)

# Generate predictions
test_texts = test_df_processed['Title']
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    outputs = trained_model(**{k: v.to(trained_model.device) for k, v in test_encodings.items()})
    predictions = outputs.logits.argmax(dim=-1).cpu().numpy()

# Map predictions back to labels
id_to_label = {idx: label for label, idx in label_to_id.items()}
predicted_labels = [id_to_label[pred] for pred in predictions]

# Build submission DataFrame
submission_df = pd.DataFrame({
    'Id': test_df_processed.index,  # Assuming there's an index or ID column
    'Domain': predicted_labels
})

# Save submission file
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission file saved at: {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
