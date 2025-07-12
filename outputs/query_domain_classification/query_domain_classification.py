import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import json
import joblib
import numpy as np

# --- File Path Constants ---
# Determine the project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_PATH_OPTION1 = (ROOT_DIR / 'input/Datasets/datasets/query_domain_classification').resolve()
BASE_PATH_OPTION2 = Path('input/Datasets/datasets/query_domain_classification').resolve()

if BASE_PATH_OPTION1.exists():
    BASE_PATH = BASE_PATH_OPTION1
elif BASE_PATH_OPTION2.exists():
    BASE_PATH = BASE_PATH_OPTION2
else:
    raise FileNotFoundError(f"Dataset base path not found at {BASE_PATH_OPTION1} or {BASE_PATH_OPTION2}")

print(f"Resolved BASE_PATH: {BASE_PATH}")

TRAIN_FILE = BASE_PATH / "train.csv"
TEST_FILE = BASE_PATH / "test.csv"
SUBMISSION_SAMPLE_FILE = BASE_PATH / "Submission_file01.csv"

# Output paths
OUTPUTS_DIR = Path("./outputs")
MODELS_DIR = Path("./models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "query_domain_classification"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / f"{DATASET_NAME}_model.pkl"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

def build_preprocessor():
    """
    Builds and returns the ColumnTransformer preprocessor.
    """
    numerical_features = []
    categorical_features = []
    text_features = ['Title']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('tfidf', TfidfVectorizer(max_features=5000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_model():
    """
    Orchestrates the data loading, cleaning, preprocessing, model training,
    evaluation, and persistence steps for the query_domain_classification dataset.
    """
    # Load datasets
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the dataset files are in the correct directory.")
        return None, None, None, None

    print("\n--- Original Train DataFrame Info ---")
    train_df.info()
    print("\n--- Original Train DataFrame Head ---")
    print(train_df.head())

    # --- Preprocessing Steps ---

    # Identify target column
    TARGET_COLUMN = "Domain"

    # Separate features (X) and target (y)
    X = train_df.drop(columns=[TARGET_COLUMN, 'ID']) # ID is not a feature for classification
    y = train_df[TARGET_COLUMN]
    X_test_final = test_df.drop(columns=['ID']) # Assuming test_df also has an 'ID' column

    # Identify column types based on metadata and EDA
    numerical_features = []
    categorical_features = []
    text_features = ['Title']

    preprocessor = build_preprocessor()

    # Apply preprocessing to the full training data before splitting
    print("\n--- Applying Preprocessing to full dataset ---")
    X['Title'] = X['Title'].fillna('').astype(str)
    X_test_final['Title'] = X_test_final['Title'].fillna('').astype(str)

    X_processed = preprocessor.fit_transform(X)
    X_test_final_processed = preprocessor.transform(X_test_final)

    print(f"Shape of full processed training data: {X_processed.shape}")
    print(f"Shape of processed final test data: {X_test_final_processed.shape}")

    # --- Stratified Train-Validation Split ---
    # Convert target to numerical labels for LightGBM
    unique_domains = y.unique()
    unique_domains.sort()
    label_mapping = {domain: i for i, domain in enumerate(unique_domains)}
    y_encoded = y.map(label_mapping)

    # Perform 80/20 stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nShape of training split: {X_train.shape}")
    print(f"Shape of validation split: {X_val.shape}")

    # --- Model Building and Training (LightGBM for Multi-class Classification) ---
    print("\n--- Training LightGBM Model ---")

    num_classes = len(unique_domains)

    lgb_clf = lgb.LGBMClassifier(objective='multiclass',
                                 num_class=num_classes,
                                 random_state=42,
                                 n_estimators=1000)

    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

    lgb_clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=callbacks)

    trained_model = lgb_clf
    print("\nModel training complete.")

    # --- Evaluation ---
    print("\n--- Evaluating Model ---")
    y_pred_val = trained_model.predict(X_val)
    y_proba_val = trained_model.predict_proba(X_val)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_val, y_pred_val)
    metrics['f1_macro'] = f1_score(y_val, y_pred_val, average='macro')
    metrics['log_loss'] = log_loss(y_val, y_proba_val)

    if num_classes > 1:
        try:
            metrics['roc_auc_ovr_macro'] = roc_auc_score(y_val, y_proba_val, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Could not calculate ROC AUC: {e}. This might happen if a class has only one sample in the validation set.")
            metrics['roc_auc_ovr_macro'] = np.nan
    else:
        metrics['roc_auc_ovr_macro'] = np.nan

    print("\n--- Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # --- Persist Trained Model ---
    joblib.dump(trained_model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    return trained_model, preprocessor, label_mapping, X_test_final_processed

def generate_predictions(trained_model=None):
    """
    Generates predictions on the test set and creates a submission file.

    Args:
        trained_model: The trained model object. If None, the model will be loaded from disk.
    """
    if trained_model is None:
        try:
            trained_model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: Model not found at {MODEL_PATH}. Please ensure the training script was run and model saved.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        print("Train and Test datasets loaded successfully for preprocessing.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the dataset files are in the correct directory.")
        return

    preprocessor = build_preprocessor()

    TARGET_COLUMN = "Domain"
    X_train_for_preprocessor_fit = train_df.drop(columns=[TARGET_COLUMN, 'ID'])
    X_test_final = test_df.drop(columns=['ID'])

    X_train_for_preprocessor_fit['Title'] = X_train_for_preprocessor_fit['Title'].fillna('').astype(str)
    X_test_final['Title'] = X_test_final['Title'].fillna('').astype(str)

    print("\n--- Applying Preprocessing to test dataset ---")
    preprocessor.fit(X_train_for_preprocessor_fit)
    X_test_final_processed = preprocessor.transform(X_test_final)

    print(f"Shape of processed test data for prediction: {X_test_final_processed.shape}")

    y_pred_encoded = trained_model.predict(X_test_final_processed)

    y_train_original = train_df[TARGET_COLUMN]
    unique_domains = y_train_original.unique()
    unique_domains.sort()
    label_mapping = {domain: i for i, domain in enumerate(unique_domains)}
    inverse_label_mapping = {i: domain for domain, i in label_mapping.items()}

    y_pred_labels = pd.Series(y_pred_encoded).map(inverse_label_mapping)

    submission_df = pd.DataFrame({'ID': test_df['ID'], 'Domain': y_pred_labels})

    try:
        sample_submission_df = pd.read_csv(SUBMISSION_SAMPLE_FILE)
        if not submission_df.columns.equals(sample_submission_df.columns):
            print("Warning: Submission columns do not match sample submission columns.")
            print(f"Expected: {sample_submission_df.columns.tolist()}")
            print(f"Got: {submission_df.columns.tolist()}")
        if not submission_df['ID'].equals(sample_submission_df['ID']):
            print("Warning: Submission 'ID' column order or values do not match sample submission 'ID'.")
            submission_df = submission_df.set_index('ID').reindex(sample_submission_df['ID']).reset_index()
            print("Submission DataFrame reindexed to match sample submission ID order.")

    except FileNotFoundError:
        print(f"Sample submission file not found at {SUBMISSION_SAMPLE_FILE}. Proceeding with default submission format.")
    except Exception as e:
        print(f"Error processing sample submission file: {e}")

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission file generated successfully at {SUBMISSION_PATH}")

def main():
    """
    Main function to orchestrate the entire workflow: training and prediction.
    """
    # Train the model
    trained_model, preprocessor, label_mapping, X_test_final_processed = train_model()

    # If training was successful, proceed to generate predictions
    if trained_model is not None:
        generate_predictions(trained_model=trained_model)
    else:
        print("Model training failed, skipping prediction generation.")

if __name__ == "__main__":
    main()