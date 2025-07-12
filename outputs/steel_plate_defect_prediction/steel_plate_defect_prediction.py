import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import json
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Define file paths (constants at the top)
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_PATH_OPTION1 = (ROOT_DIR / 'input/Datasets/datasets/steel_plate_defect_prediction').resolve()
BASE_PATH_OPTION2 = Path('input/Datasets/datasets/steel_plate_defect_prediction').resolve()

if BASE_PATH_OPTION1.exists():
    BASE_PATH = BASE_PATH_OPTION1
elif BASE_PATH_OPTION2.exists():
    BASE_PATH = BASE_PATH_OPTION2
else:
    raise FileNotFoundError(f"Dataset base path not found at {BASE_PATH_OPTION1} or {BASE_PATH_OPTION2}")

print(f"Resolved BASE_PATH: {BASE_PATH}")

TRAIN_FILE = BASE_PATH / 'train.csv'
TEST_FILE = BASE_PATH / 'test.csv'
SAMPLE_SUBMISSION_FILE = BASE_PATH / 'sample_submission.csv'

# Output paths
OUTPUTS_DIR = Path("./outputs")
MODELS_DIR = Path("./models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUTS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "steel_plate_defect_prediction_model.pkl"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# Paths for preprocessors saved in Stage 1
SCALER_PATH = MODELS_DIR / "scaler.pkl"
NUMERICAL_IMPUTER_PATH = MODELS_DIR / "numerical_imputer.pkl"

# Dataset Metadata
TARGET_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
]
ID_COLUMN = 'id'

def preprocess_data(is_training=True):
    """
    Orchestrates the data loading, cleaning, and preprocessing steps for the
    steel plate defect prediction dataset. This function is consolidated
    from the original `preprocess_data_for_training` and `preprocess_data_for_prediction`.

    Args:
        is_training (bool): If True, loads train.csv and fits preprocessors.
                            If False, loads test.csv and transforms using saved preprocessors.
    Returns:
        tuple: (X_processed, y_train_or_test_ids, TARGET_COLUMNS)
               y_train_or_test_ids will be y_train (DataFrame) if is_training=True,
               or test_ids (Series) if is_training=False.
    """
    if is_training:
        print(f"Loading train data from: {TRAIN_FILE}")
        df = pd.read_csv(TRAIN_FILE)
        X = df.drop(columns=TARGET_COLUMNS + [ID_COLUMN])
        y = df[TARGET_COLUMNS]
    else:
        print(f"Loading test data from: {TEST_FILE}")
        df = pd.read_csv(TEST_FILE)
        X = df.drop(columns=[ID_COLUMN])
        y = df[ID_COLUMN] # For test set, y will be the IDs

    # Identify column types
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    print(f"Identified numerical columns for preprocessing: {numerical_cols}")

    # --- Preprocessing Steps ---
    if is_training:
        print("Processing training data: Fitting and transforming imputer and scaler.")
        numerical_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
        joblib.dump(numerical_imputer, NUMERICAL_IMPUTER_PATH)

        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        joblib.dump(scaler, SCALER_PATH)
    else:
        print("Processing test data: Loading and transforming with saved imputer and scaler.")
        try:
            numerical_imputer = joblib.load(NUMERICAL_IMPUTER_PATH)
            scaler = joblib.load(SCALER_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Preprocessors not found. Ensure training mode was run and saved them. Error: {e}")

        X[numerical_cols] = numerical_imputer.transform(X[numerical_cols])
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    print("\nPreprocessing complete.")
    print(f"Shape of preprocessed X: {X.shape}")
    if is_training:
        print(f"Shape of y: {y.shape}")
    else:
        print(f"Shape of test_ids: {y.shape}")

    # Display first few rows of preprocessed data
    print("\nPreprocessed X head:")
    print(X.head())
    if is_training:
        print("\ny head:")
        print(y.head())

    return X, y, TARGET_COLUMNS

def train_and_evaluate_model(X_train_full, y_train_full, X_test, test_ids):
    """
    Splits data, trains a LightGBM model for multi-label classification,
    evaluates it, and persists metrics and the model.
    """
    # Use MultilabelStratifiedKFold for the train-validation split to handle multi-label stratification.
    # We'll use a single split (n_splits=5, taking the first one for approx 80/20 split)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

    # Get the indices for the first split (approx 80% train, 20% validation)
    train_index, val_index = next(mskf.split(X_train_full, y_train_full))

    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

    print(f"\nShape of X_train after stratified split: {X_train.shape}")
    print(f"Shape of y_train after stratified split: {y_train.shape}")
    print(f"Shape of X_val after stratified split: {X_val.shape}")
    print(f"Shape of y_val after stratified split: {y_val.shape}")

    trained_models = {}
    metrics = {"overall": {}, "per_target": {}}

    val_preds = pd.DataFrame(index=y_val.index, columns=TARGET_COLUMNS)
    test_preds = pd.DataFrame(index=X_test.index, columns=TARGET_COLUMNS)

    for target_col in TARGET_COLUMNS:
        print(f"\nTraining model for target: {target_col}")

        # LightGBM Classifier for binary classification
        lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=1000, n_jobs=-1)

        # Check if the target column in the training split has only one class
        if y_train[target_col].nunique() < 2:
            print(f"  Skipping training for {target_col}: Only one class present in training split. Assigning default predictions.")
            # Assign default probabilities (e.g., 0.5) or the majority class probability
            val_preds[target_col] = 0.5
            test_preds[target_col] = 0.5

            # Calculate metrics for this skipped target. ROC AUC is undefined, set to 0.5.
            majority_class = y_train[target_col].mode()[0]
            metrics["per_target"][target_col] = {
                "roc_auc": 0.5, 
                "accuracy": accuracy_score(y_val[target_col], np.full_like(y_val[target_col], majority_class)),
                "f1_score": f1_score(y_val[target_col], np.full_like(y_val[target_col], majority_class)),
                "log_loss": log_loss(y_val[target_col], np.full_like(y_val[target_col], 0.5))
            }
            trained_models[target_col] = None # Mark as not trained
            continue # Skip to next target

        lgb_clf.fit(X_train, y_train[target_col],
                    eval_set=[(X_val, y_val[target_col])],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(100, verbose=False)])

        trained_models[target_col] = lgb_clf

        # Evaluate on validation set
        y_val_pred_proba = lgb_clf.predict_proba(X_val)[:, 1]
        y_val_pred_class = (y_val_pred_proba > 0.5).astype(int)

        val_preds[target_col] = y_val_pred_proba

        # Calculate metrics for current target
        roc_auc = roc_auc_score(y_val[target_col], y_val_pred_proba)
        accuracy = accuracy_score(y_val[target_col], y_val_pred_class)
        f1 = f1_score(y_val[target_col], y_val_pred_class)
        logloss = log_loss(y_val[target_col], y_val_pred_proba)

        metrics["per_target"][target_col] = {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "f1_score": f1,
            "log_loss": logloss
        }
        print(f"  {target_col} - ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, LogLoss: {logloss:.4f}")

        # Predict on test set
        test_preds[target_col] = lgb_clf.predict_proba(X_test)[:, 1]

    # Calculate overall metrics (average ROC AUC as per task definition)
    trained_target_metrics_roc_auc = [metrics["per_target"][col]["roc_auc"] for col in TARGET_COLUMNS if trained_models[col] is not None]
    if trained_target_metrics_roc_auc:
        overall_roc_auc = np.mean(trained_target_metrics_roc_auc)
    else:
        overall_roc_auc = 0.0 # If no models were trained, set to 0.0

    metrics["overall"]["average_roc_auc"] = overall_roc_auc

    overall_accuracy = np.mean([metrics["per_target"][col]["accuracy"] for col in TARGET_COLUMNS])
    overall_f1 = np.mean([metrics["per_target"][col]["f1_score"] for col in TARGET_COLUMNS])
    overall_logloss = np.mean([metrics["per_target"][col]["log_loss"] for col in TARGET_COLUMNS])

    metrics["overall"]["average_accuracy"] = overall_accuracy
    metrics["overall"]["average_f1_score"] = overall_f1
    metrics["overall"]["average_log_loss"] = overall_logloss

    print(f"\nOverall Average ROC AUC: {overall_roc_auc:.4f}")
    print(f"Overall Average Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Average F1 Score: {overall_f1:.4f}")
    print(f"Overall Average Log Loss: {overall_logloss:.4f}")

    # Persist metrics to JSON file
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Persist the trained models (dictionary of models)
    joblib.dump(trained_models, MODEL_PATH)
    print(f"Trained models saved to {MODEL_PATH}")

    return trained_models, test_preds

def generate_predictions(X_test_processed, test_ids, trained_models=None):
    """
    Generates predictions using the trained models.

    Args:
        X_test_processed (pd.DataFrame): Preprocessed test features.
        test_ids (pd.Series): Original 'id' column from the test set.
        trained_models (dict, optional): Dictionary of trained models.
                                         If None, models will be loaded from MODEL_PATH.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and probability predictions for each target.
    """
    # 1. Ensure trained_models is available
    if trained_models is None:
        print(f"Loading trained models from {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained models not found at {MODEL_PATH}. Please run the training script first.")
        trained_models = joblib.load(MODEL_PATH)
        print("Models loaded successfully.")

    # 2. Generate predictions
    # Initialize a DataFrame to store predictions
    test_predictions = pd.DataFrame(index=X_test_processed.index, columns=TARGET_COLUMNS)

    for target_col in TARGET_COLUMNS:
        # Check if the model for the current target exists and is not None (i.e., was successfully trained)
        if target_col in trained_models and trained_models[target_col] is not None:
            model = trained_models[target_col]
            # Classification task, so we need predict_proba for ROC AUC metric
            # Ensure the model has predict_proba method
            if hasattr(model, 'predict_proba'):
                test_predictions[target_col] = model.predict_proba(X_test_processed)[:, 1]
            else:
                # Fallback if a model somehow doesn't have predict_proba (shouldn't happen with LGBM)
                print(f"Warning: Model for {target_col} does not have predict_proba. Assigning default 0.5.")
                test_predictions[target_col] = 0.5
        else:
            # This handles cases where a model was explicitly set to None in Stage 2
            # because the training data for that specific target had only one class.
            print(f"Warning: Model for {target_col} was not trained (or is None). Assigning default 0.5.")
            test_predictions[target_col] = 0.5 # Assign a default value if model is missing or None

    # 3. Build submission_df following the sample submission format
    submission_df = pd.DataFrame({'id': test_ids})
    for col in TARGET_COLUMNS:
        submission_df[col] = test_predictions[col]

    # 4. Save the submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    # 5. Print a short confirmation message
    print(f"\nSubmission file generated at {SUBMISSION_PATH}")

    return submission_df

def main():
    # Stage 1 & 2: Preprocess training data and train model
    X_train_processed, y_train_processed, _ = preprocess_data(is_training=True)

    # Preprocess test data for prediction
    X_test_processed, test_ids, _ = preprocess_data(is_training=False)

    # Train and evaluate model, and get test predictions
    trained_models, test_predictions = train_and_evaluate_model(X_train_processed, y_train_processed, X_test_processed, test_ids)
    
    # Stage 3: Generate final submission file
    generate_predictions(X_test_processed, test_ids, trained_models)

if __name__ == '__main__':
    main()