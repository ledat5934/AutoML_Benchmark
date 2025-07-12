import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import json
import joblib
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- File Path Constants (Overrideable) ---
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_PATH_OPTION1 = (ROOT_DIR / 'input/Datasets/datasets/predict_the_llm').resolve()
BASE_PATH_OPTION2 = Path('input/Datasets/datasets/predict_the_llm').resolve()

if BASE_PATH_OPTION1.exists():
    BASE_PATH = BASE_PATH_OPTION1
elif BASE_PATH_OPTION2.exists():
    BASE_PATH = BASE_PATH_OPTION2
else:
    raise FileNotFoundError(f"Could not find the dataset base path. Tried: {BASE_PATH_OPTION1} and {BASE_PATH_OPTION2}")

print(f"Resolved BASE_PATH: {BASE_PATH}")

TRAIN_FILE = BASE_PATH / "train.csv"
TEST_FILE = BASE_PATH / "test.csv"
SAMPLE_SUBMISSION_FILE = BASE_PATH / "sample_submission.csv"

OUTPUTS_DIR = Path("./outputs")
MODELS_DIR = Path("./models")
PROCESSED_DIR = Path("./processed") # New directory for processed data
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "predict_the_llm_model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# Custom transformer to ensure string type for text columns
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Ensure X is converted to a NumPy array first, then to string type, then reshape to 1D.
        # This handles cases where X might be a pandas Series or a NumPy array.
        return np.asarray(X).astype(str).reshape(-1)

def main():
    """
    Orchestrates the data loading, preprocessing, splitting, model training,
    evaluation, and persistence for the predict_the_llm dataset.
    """
    # Load metadata (assuming it's provided as a string or loaded from a file)
    metadata = {
      "dataset_info": {
        "name": "predict_the_llm",
        "base_path": "input/Datasets/datasets/predict_the_llm",
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
          "n": 3180,
          "n_var": 3,
          "memory_size": 76448,
          "record_size": 24.040251572327044,
          "n_cells_missing": 6,
          "p_cells_missing": 0.0006289308176100629,
          "size_optimized": True,
          "optimization_level": "aggressive",
          "optimization_note": "All value lists removed - only counts and basic statistics retained",
          "removed_sections": 42,
          "optimization_strategy": "Minimal JSON for maximum compatibility with LLM token limits"
        },
        "variables": {
          "Question": {
            "n_distinct": 568,
            "p_distinct": 0.17861635220125785,
            "is_unique": False,
            "n_unique": 1,
            "p_unique": 0.00031446540880503143,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 3180,
            "p_missing": 0.0,
            "count": 3180,
            "memory_size": 25568,
            "max_length": 229,
            "mean_length": 56.09088050314465,
            "median_length": 93,
            "min_length": 16,
            "n_characters_distinct": 86,
            "n_characters": 178369,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "Response": {
            "n_distinct": 3173,
            "p_distinct": 0.9996849401386263,
            "is_unique": False,
            "n_unique": 3172,
            "p_unique": 0.9993698802772527,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 6,
            "n": 3180,
            "p_missing": 0.0018867924528301887,
            "count": 3174,
            "memory_size": 25568,
            "max_length": 3878,
            "mean_length": 859.661940768746,
            "median_length": 1784,
            "min_length": 1,
            "n_characters_distinct": 159,
            "n_characters": 2728567,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "target": {
            "n_distinct": 7,
            "p_distinct": 0.00220125786163522,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 3180,
            "p_missing": 0.0,
            "count": 3180,
            "memory_size": 25568,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 455,
            "mean": 2.998427672955975,
            "std": 2.0007070265197857,
            "variance": 4.002828605965643,
            "min": 0,
            "max": 6,
            "kurtosis": -1.2506476076132633,
            "skewness": 0.000983515718696273,
            "sum": 9535,
            "mad": 2.0,
            "range": 6,
            "5%": 0.0,
            "25%": 1.0,
            "50%": 3.0,
            "75%": 5.0,
            "95%": 6.0,
            "iqr": 4.0,
            "cv": 0.6672520549903428,
            "p_zeros": 0.1430817610062893,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          }
        },
        "scatter": {},
        "correlations": {}
      },
      "task_definition": {
        "description_summary": "The dataset is for a competition to identify which of 7 possible LLM models generated a given text response. Participants need to predict the probability for each of the 7 models for every response in the test set.",
        "task_type": "multi_class_classification",
        "target_columns": [
          "target"
        ],
        "evaluation_metric": "logloss"
      }
    }

    # Load datasets
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure the dataset files are in the correct location.")
        return None # Return None if files are not found

    print("\nOriginal Train DataFrame Info:")
    train_df.info()
    print("\nOriginal Test DataFrame Info:")
    test_df.info()

    # Identify target column
    target_column = metadata['task_definition']['target_columns'][0]

    # Separate features and target
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    X_test_raw = test_df.copy() # Keep original test_df for transformation

    # Identify column types based on metadata
    numerical_cols = []
    categorical_cols = []
    text_cols = []

    for col, info in metadata['profiling_summary']['variables'].items():
        if col == target_column:
            continue
        if info['type'] == 'Numeric':
            numerical_cols.append(col)
        elif info['type'] == 'Text':
            text_cols.append(col)

    print(f"\nIdentified Numerical Columns: {numerical_cols}")
    print(f"Identified Categorical Columns: {categorical_cols}")
    print(f"Identified Text Columns: {text_cols}")

    # Preprocessing Pipelines
    # Each text column ('Question', 'Response') will be processed by its own TF-IDF vectorizer.
    # ColumnTransformer will then concatenate these sparse outputs horizontally.
    # This is the standard and correct way to handle multiple text columns.
    # The TextCleaner ensures the input to TfidfVectorizer is always a 1D array of strings.

    # Create a list of (name, transformer, columns) tuples for ColumnTransformer
    transformers_list = []

    for col in text_cols:
        text_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('text_cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
        ])
        # Pass each text column individually to its own TF-IDF pipeline
        transformers_list.append((f'text_{col}', text_pipeline, [col])) # Note: [col] to ensure ColumnTransformer passes a Series/DataFrame column

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough' # Keep other columns (e.g., IDs if present)
    )

    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Split data into training and validation sets
    print("\nSplitting data into training and validation sets (80/20 stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Fit and transform the training data
    print("\nFitting and transforming training data...")
    # Ensure X_train is a DataFrame when passed to fit_transform
    X_train_processed = full_pipeline.fit_transform(X_train)
    print("Training data preprocessing complete.")

    # Transform the validation data
    print("Transforming validation data...")
    # Ensure X_val is a DataFrame when passed to transform
    X_val_processed = full_pipeline.transform(X_val)
    print("Validation data preprocessing complete.")

    # Transform the test data
    print("Transforming test data...")
    # Ensure X_test_raw is a DataFrame when passed to transform
    X_test_processed = full_pipeline.transform(X_test_raw)
    print("Test data preprocessing complete.")

    # Display shapes of processed data
    print(f"\nShape of processed training features: {X_train_processed.shape}")
    print(f"Shape of processed validation features: {X_val_processed.shape}")
    print(f"Shape of processed test features: {X_test_processed.shape}")

    # --- Model Training ---
    print("\nStarting model training (LightGBM Classifier)...")
    num_classes = y.nunique()

    lgb_clf = lgb.LGBMClassifier(objective='multiclass',
                                 num_class=num_classes,
                                 random_state=42,
                                 n_estimators=1000,
                                 learning_rate=0.05)

    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

    lgb_clf.fit(X_train_processed, y_train,
                eval_set=[(X_val_processed, y_val)],
                eval_metric='multi_logloss',
                callbacks=callbacks)

    trained_model = lgb_clf
    print("Model training complete.")

    # --- Evaluation ---
    print("\nEvaluating model performance on the validation set...")
    y_pred_proba = trained_model.predict_proba(X_val_processed)
    y_pred = trained_model.predict(X_val_processed)

    metrics = {}
    metrics['logloss'] = log_loss(y_val, y_pred_proba)
    metrics['accuracy'] = accuracy_score(y_val, y_pred)
    metrics['f1_macro'] = f1_score(y_val, y_pred, average='macro')

    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
    except ValueError as e:
        print(f"Could not calculate ROC AUC (ovr): {e}. This might happen if a class has only one sample in y_val.")
        metrics['roc_auc_ovr'] = None

    print(f"Validation LogLoss: {metrics['logloss']:.4f}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 (Macro): {metrics['f1_macro']:.4f}")
    if metrics['roc_auc_ovr'] is not None:
        print(f"Validation ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    else:
        print("Validation ROC AUC (OvR): Not calculated due to error.")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # --- Model Persistence ---
    joblib.dump(trained_model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    # --- Prediction and Submission Generation ---
    print("\nGenerating predictions for the test set...")
    # Ensure trained_model is available. If main() was called and returned None, load it.
    if trained_model is None:
        try:
            trained_model = joblib.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}. Cannot generate predictions.")
            return

    # Generate probabilities for the test set
    test_predictions_proba = trained_model.predict_proba(X_test_processed)

    # Prepare submission DataFrame
    # The sample submission has columns 'id', '0', '1', ..., '6'
    # Check if 'id' column exists in test_df, if not, create a default index
    if 'id' in test_df.columns:
        submission_df = pd.DataFrame({'id': test_df['id']})
    else:
        # If 'id' column is not present, assume the test_df index should be used
        # or a simple range if no meaningful index exists.
        # The sample submission implies an 'id' column, so we should create one if missing.
        submission_df = pd.DataFrame({'id': test_df.index}) 

    # Add probability columns
    # The target column 'target' has values from 0 to 6, so we need 7 probability columns.
    # The order of classes in predict_proba is determined by the model's internal class mapping.
    # For LightGBM, it's usually sorted unique values of y_train.
    # To be safe, we can get the class labels from the trained model.
    class_labels = trained_model.classes_
    for i, class_label in enumerate(class_labels):
        submission_df[str(class_label)] = test_predictions_proba[:, i]

    # Save submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file generated and saved to {SUBMISSION_PATH}")
    print(f"Submission DataFrame head:\n{submission_df.head()}")

    return trained_model

if __name__ == "__main__":
    trained_model_instance = main()
    if trained_model_instance:
        print("\nScript finished successfully. Trained model instance returned.")