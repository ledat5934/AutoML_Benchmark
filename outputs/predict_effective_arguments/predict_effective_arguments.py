import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import numpy as np
import json
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

# Define constants for file paths
ROOT_DIR = Path(__file__).resolve().parent.parent

# Option 1: Relative to project root (for local development/testing)
BASE_PATH_OPTION1 = (ROOT_DIR / 'input/Datasets/datasets/predict_effective_arguments').resolve()
# Option 2: Relative to current working directory (for Kaggle or specific environments)
BASE_PATH_OPTION2 = Path('input/Datasets/datasets/predict_effective_arguments').resolve()

# Determine the actual base path
if BASE_PATH_OPTION1.exists():
    BASE_PATH = BASE_PATH_OPTION1
else:
    BASE_PATH = BASE_PATH_OPTION2

print(f"Resolved BASE_PATH: {BASE_PATH}")

# Define model and metrics paths
MODEL_PATH = Path("./models/predict_effective_arguments_model.pkl").resolve()
METRICS_PATH = Path("./outputs/metrics.json").resolve()
LABEL_ENCODER_PATH = Path("./models/label_encoder.pkl").resolve() # Path to save LabelEncoder
SUBMISSION_PATH = Path("./outputs/submission.csv").resolve()

# Ensure output directories exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
LABEL_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_dataset_metadata():
    """
    Returns the hardcoded dataset metadata.
    """
    return {
      "dataset_info": {
        "name": "predict_effective_arguments",
        "base_path": "input/Datasets/datasets/predict_effective_arguments",
        "description_file": "description.txt",
        "files": [
          {
            "path": "sample_submission.csv",
            "role": "sample",
            "type": "tabular"
          },
          {
            "path": "test.csv",
            "role": "test",
            "type": "tabular"
          },
          {
            "path": "train.csv",
            "role": "train",
            "type": "tabular"
          }
        ]
      },
      "profiling_summary": {
        "time_index_analysis": "None",
        "table": {
          "n": 29574,
          "n_var": 5,
          "memory_size": 1183088,
          "record_size": 40.00432812605667,
          "n_cells_missing": 0,
          "p_cells_missing": 0.0,
          "size_optimized": True,
          "optimization_level": "aggressive",
          "optimization_note": "All value lists removed - only counts and basic statistics retained",
          "removed_sections": 84,
          "optimization_strategy": "Minimal JSON for maximum compatibility with LLM token limits"
        },
        "variables": {
          "discourse_id": {
            "n_distinct": 29574,
            "p_distinct": 1.0,
            "is_unique": True,
            "n_unique": 29574,
            "p_unique": 1.0,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 29574,
            "p_missing": 0.0,
            "count": 29574,
            "memory_size": 236720,
            "max_length": 12,
            "mean_length": 12.0,
            "median_length": 12,
            "min_length": 12,
            "n_characters_distinct": 16,
            "n_characters": 354888,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "essay_id": {
            "n_distinct": 3352,
            "p_distinct": 0.1133428011090823,
            "is_unique": False,
            "n_unique": 59,
            "p_unique": 0.0019949956042469735,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 29574,
            "p_missing": 0.0,
            "count": 29574,
            "memory_size": 236720,
            "max_length": 12,
            "mean_length": 12.0,
            "median_length": 12,
            "min_length": 12,
            "n_characters_distinct": 16,
            "n_characters": 354888,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "discourse_text": {
            "n_distinct": 29520,
            "p_distinct": 0.9981740718198417,
            "is_unique": False,
            "n_unique": 29486,
            "p_unique": 0.9970244133360384,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 29574,
            "p_missing": 0.0,
            "count": 29574,
            "memory_size": 236720,
            "max_length": 3808,
            "mean_length": 249.7570839250693,
            "median_length": 1467,
            "min_length": 4,
            "n_characters_distinct": 108,
            "n_characters": 7386316,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "discourse_type": {
            "n_distinct": 7,
            "p_distinct": 0.0002366943937242172,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 29574,
            "p_missing": 0.0,
            "count": 29574,
            "memory_size": 236720,
            "max_length": 20,
            "mean_length": 8.06732264827213,
            "median_length": 12,
            "min_length": 4,
            "n_characters_distinct": 23,
            "n_characters": 238583,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "discourse_effectiveness": {
            "n_distinct": 3,
            "p_distinct": 0.00010144045445323595,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 29574,
            "p_missing": 0.0,
            "count": 29574,
            "memory_size": 236720,
            "max_length": 11,
            "mean_length": 8.780651923987286,
            "median_length": 8,
            "min_length": 8,
            "n_characters_distinct": 14,
            "n_characters": 259679,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          }
        },
        "scatter": {},
        "correlations": {}
      },
      "task_definition": {
        "description_summary": "The dataset contains argumentative essays from 6th-12th grade US students, with individual discourse elements (e.g., claims, evidence) annotated by expert raters. The goal is to classify each discourse element's quality as 'effective', 'adequate', or 'ineffective'. The competition has two tracks: one focused on classification accuracy and another on computational efficiency combined with accuracy.",
        "task_type": "multi_class_classification",
        "target_columns": [
          "discourse_effectiveness"
        ],
        "evaluation_metric": "multi-class logarithmic loss"
      }
    }

def train_model():
    """
    Orchestrates the data loading, preprocessing, splitting, model training,
    evaluation, and persistence for the predict_effective_arguments dataset.
    """
    dataset_metadata = get_dataset_metadata()

    # Get file paths from metadata
    train_file_info = next(f for f in dataset_metadata['dataset_info']['files'] if f['role'] == 'train')
    test_file_info = next(f for f in dataset_metadata['dataset_info']['files'] if f['role'] == 'test')

    train_path = BASE_PATH / train_file_info['path']
    test_path = BASE_PATH / test_file_info['path']

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\n--- Original Training Data Info ---")
    train_df.info()
    print("\n--- Original Training Data Head ---")
    print(train_df.head())

    # Identify column types based on metadata and common sense
    target_column = dataset_metadata['task_definition']['target_columns'][0]

    # Features are all columns except discourse_id (identifier) and the target
    features = [col for col in train_df.columns if col not in ['discourse_id', target_column]]

    # Separate features by type
    numerical_cols = [] # No explicit numerical columns in this dataset based on metadata
    categorical_cols = []
    text_cols = []

    for col in features:
        var_info = dataset_metadata['profiling_summary']['variables'].get(col)
        if var_info:
            if var_info['type'] == 'Text':
                if col == 'discourse_text':
                    text_cols.append(col)
                else: # discourse_type, essay_id are categorical text
                    categorical_cols.append(col)
            # Add logic for numerical if any were present
        else:
            # Fallback if column not explicitly in metadata variables (e.g., if added later)
            if train_df[col].dtype == 'object':
                # Heuristic: if text, check for high cardinality or length
                if train_df[col].nunique() > 500 or train_df[col].apply(lambda x: len(str(x))).mean() > 50:
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(train_df[col]):
                numerical_cols.append(col)

    print(f"\nIdentified Numerical Columns: {numerical_cols}")
    print(f"Identified Categorical Columns: {categorical_cols}")
    print(f"Identified Text Columns: {text_cols}")
    print(f"Target Column: {target_column}")

    # Preprocessing Pipelines
    preprocessor_steps = []

    # Categorical Pipeline
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        preprocessor_steps.append(('cat', categorical_transformer, categorical_cols))

    # Text Pipeline (TF-IDF)
    if text_cols:
        text_transformer = TfidfVectorizer(stop_words='english', max_features=5000)
        preprocessor_steps.append(('text_tfidf', text_transformer, text_cols))

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=preprocessor_steps,
        remainder='passthrough'
    )

    # Separate target variable
    X = train_df.drop(columns=[target_column, 'discourse_id'])
    y = train_df[target_column]

    # Encode target variable for LightGBM (requires integer labels for multi-class)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Target classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")

    # Save the label encoder for consistent inverse transformation in Stage 3
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"LabelEncoder saved to {LABEL_ENCODER_PATH}")

    # 80/20 stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nShape of X_train before preprocessing: {X_train.shape}")
    print(f"Shape of X_val before preprocessing: {X_val.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"\nShape of processed training data: {X_train_processed.shape}")
    print(f"Shape of processed validation data: {X_val_processed.shape}")

    print("\n--- Preprocessing Complete ---")

    # Build and train the model (LightGBM for multi-class classification)
    print("\n--- Training LightGBM Model ---")

    # LightGBM parameters for multi-class classification
    lgb_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    model = lgb.LGBMClassifier(**lgb_params)

    # Fit the model with early stopping
    model.fit(X_train_processed, y_train,
              eval_set=[(X_val_processed, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    print("\n--- Model Training Complete ---")

    # Evaluate the model on the validation set
    print("\n--- Evaluating Model ---")
    y_pred_proba = model.predict_proba(X_val_processed)
    y_pred = model.predict(X_val_processed)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_val, y_pred)
    metrics['f1_macro'] = f1_score(y_val, y_pred, average='macro')
    metrics['logloss'] = log_loss(y_val, y_pred_proba)
    metrics['roc_auc_ovr'] = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')

    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"Validation LogLoss: {metrics['logloss']:.4f}")
    print(f"Validation ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")

    # Persist metrics to JSON file
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Persist the trained model
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])

    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    return full_pipeline, label_encoder.classes_

def generate_predictions(trained_model=None):
    """
    Loads the test data, preprocesses it, generates predictions using the trained model,
    and saves the predictions in the specified submission format.

    Args:
        trained_model: An optional pre-trained model pipeline. If None, the model
                       will be loaded from MODEL_PATH.
    """
    dataset_metadata = get_dataset_metadata()

    # Load the trained model if not provided
    if trained_model is None:
        print(f"Loading trained model from {MODEL_PATH}")
        try:
            trained_model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the training script has been run.")
            return
    else:
        print("Using provided trained model.")

    # Load the LabelEncoder used during training
    print(f"Loading LabelEncoder from {LABEL_ENCODER_PATH}")
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        print(f"Error: LabelEncoder file not found at {LABEL_ENCODER_PATH}. Please ensure the training script has been run and saved the LabelEncoder.")
        return

    # Get file paths from metadata
    test_file_info = next(f for f in dataset_metadata['dataset_info']['files'] if f['role'] == 'test')
    sample_submission_info = next(f for f in dataset_metadata['dataset_info']['files'] if f['role'] == 'sample')

    test_path = BASE_PATH / test_file_info['path']
    sample_submission_path = BASE_PATH / sample_submission_info['path']

    # Load test data and sample submission
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_submission_path)

    print("\n--- Test Data Info ---")
    test_df.info()
    print("\n--- Test Data Head ---")
    print(test_df.head())

    # Extract discourse_id for submission
    discourse_ids = test_df['discourse_id']

    # Make predictions
    X_test_raw = test_df.drop(columns=['discourse_id'])

    print("\n--- Generating Predictions ---")

    predictions_proba = trained_model.predict_proba(X_test_raw)

    target_classes = label_encoder.classes_
    print(f"Target classes from LabelEncoder: {target_classes}")

    # Create submission DataFrame
    submission_df = pd.DataFrame({'discourse_id': discourse_ids})

    proba_cols = [col for col in sample_submission_df.columns if col != 'discourse_id']

    class_to_index = {class_name: i for i, class_name in enumerate(target_classes)}

    for col_name in proba_cols:
        if col_name in class_to_index:
            idx = class_to_index[col_name]
            submission_df[col_name] = predictions_proba[:, idx]
        else:
            print(f"Warning: Column '{col_name}' from sample submission not found in trained model's classes. Setting to 0.0.")
            submission_df[col_name] = 0.0

    submission_df = submission_df[sample_submission_df.columns]

    # Save the submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nPredictions saved to {SUBMISSION_PATH}")
    print("\n--- Submission File Head ---")
    print(submission_df.head())

def main():
    """
    Main function to orchestrate the entire workflow: training and prediction.
    """
    print("--- Starting Model Training ---")
    trained_model_pipeline, label_encoder_classes = train_model()
    print("\n--- Starting Prediction Generation ---")
    generate_predictions(trained_model=trained_model_pipeline)

if __name__ == "__main__":
    main()