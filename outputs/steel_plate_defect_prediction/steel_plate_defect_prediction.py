import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test datasets from the specified base path.

    Args:
        base_path (str): The base directory where the dataset files are located.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"Successfully loaded train.csv from: {train_path}")
        print(f"Successfully loaded test.csv from: {test_path}")
        print("\nTrain data info:")
        train_df.info()
        print("\nTest data info:")
        test_df.info()

        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check the BASE_PATH and file names.")
        raise

def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
    """
    Performs preprocessing steps on the dataset:
    - Separates features and target columns.
    - Scales all numerical features using StandardScaler.
    - Handles the 'id' column by separating it for the test set.

    Args:
        train_df (pd.DataFrame): The raw training DataFrame.
        test_df (pd.DataFrame): The raw test DataFrame.
        target_columns (list[str]): A list of column names that represent the targets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
            - X_train_processed (pd.DataFrame): Preprocessed training features.
            - y_train (pd.DataFrame): Training target columns.
            - X_test_processed (pd.DataFrame): Preprocessed test features.
            - test_ids (pd.Series): The 'id' column from the test set.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """

    # Store 'id' column for potential submission file generation
    test_ids = test_df['id']

    # Define feature columns (all columns except 'id' and target columns)
    feature_columns = [col for col in train_df.columns if col not in target_columns + ['id']]

    # Separate features and targets for training data
    X_train = train_df[feature_columns]
    y_train = train_df[target_columns]

    # Select feature columns for test data (assuming test.csv does not contain target columns)
    X_test = test_df[feature_columns]

    # All identified features are numerical based on the dataset metadata.
    # Binary columns like 'TypeOfSteel_A300' and 'TypeOfSteel_A400' are treated as numerical
    # and will be scaled along with other numerical features.
    numerical_features = feature_columns

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training features and transform both training and test features
    X_train_scaled = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled = scaler.transform(X_test[numerical_features])

    # Convert scaled arrays back to DataFrames, preserving column names and index
    X_train_processed = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

    print("\nData preprocessing completed.")
    print(f"Shape of preprocessed training features (X_train_processed): {X_train_processed.shape}")
    print(f"Shape of training targets (y_train): {y_train.shape}")
    print(f"Shape of preprocessed test features (X_test_processed): {X_test_processed.shape}")

    return X_train_processed, y_train, X_test_processed, test_ids, scaler

def main():
    """
    Main function to orchestrate the data loading and preprocessing pipeline.
    """

    # --- File Path Constants ---
    # Default paths mirroring the standard Kaggle notebook directory layout
    KAGGLE_BASE_PATH = "./"

    # Fallback for local development or different environments
    # Adjust this path if your local dataset directory structure is different
    LOCAL_BASE_PATH = "./" 

    # Determine the actual base path based on existence
    if os.path.exists(KAGGLE_BASE_PATH):
        BASE_PATH = KAGGLE_BASE_PATH
    elif os.path.exists(LOCAL_BASE_PATH):
        BASE_PATH = LOCAL_BASE_PATH
    else:
        raise FileNotFoundError(f"Dataset base path not found at {KAGGLE_BASE_PATH} or {LOCAL_BASE_PATH}")

    # Define target columns based on the dataset metadata
    TARGET_COLUMNS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # 1. Load the dataset
    train_df, test_df = load_data(BASE_PATH)

    # 2. Perform preprocessing
    X_train_processed, y_train, X_test_processed, test_ids, scaler = preprocess_data(
        train_df.copy(), test_df.copy(), TARGET_COLUMNS
    )

    # Return the preprocessed data and scaler for subsequent ML stages
    return X_train_processed, y_train, X_test_processed, test_ids, scaler

if __name__ == "__main__":
    # Execute the main pipeline when the script is run
    X_train, y_train, X_test, test_ids, scaler = main()

    # Display a sample of the preprocessed data to verify
    print("\n--- Sample of Preprocessed Data ---")
    print("\nFirst 5 rows of preprocessed X_train:")
    print(X_train.head())
    print("\nFirst 5 rows of y_train:")
    print(y_train.head())
    print("\nFirst 5 rows of preprocessed X_test:")
    print(X_test.head())
    print("\nFirst 5 test IDs:")
    print(test_ids.head())
    print(f"\nShape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from lightgbm import LGBMClassifier
import lightgbm as lgb # Import lightgbm for early stopping callback
import joblib
import os

# --- File Path Constants ---
# Default paths mirroring the standard Kaggle notebook directory layout
KAGGLE_BASE_PATH = "./"

# Fallback for local development or different environments
# Adjust this path if your local dataset directory structure is different
LOCAL_BASE_PATH = "./" 

# Determine the actual base path based on existence
if os.path.exists(KAGGLE_BASE_PATH):
    BASE_PATH = KAGGLE_BASE_PATH
elif os.path.exists(LOCAL_BASE_PATH):
    BASE_PATH = LOCAL_BASE_PATH
else:
    raise FileNotFoundError(f"Dataset base path not found at {KAGGLE_BASE_PATH} or {LOCAL_BASE_PATH}")

# Define target columns based on the dataset metadata
TARGET_COLUMNS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# Model persistence path
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "steel_plate_defect_prediction_model.pkl")

def load_data(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test datasets from the specified base path.

    Args:
        base_path (str): The base directory where the dataset files are located.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"Successfully loaded train.csv from: {train_path}")
        print(f"Successfully loaded test.csv from: {test_path}")
        print("\nTrain data info:")
        train_df.info()
        print("\nTest data info:")
        test_df.info()

        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check the BASE_PATH and file names.")
        raise

def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
    """
    Performs preprocessing steps on the dataset:
    - Separates features and target columns.
    - Scales all numerical features using StandardScaler.
    - Handles the 'id' column by separating it for the test set.

    Args:
        train_df (pd.DataFrame): The raw training DataFrame.
        test_df (pd.DataFrame): The raw test DataFrame.
        target_columns (list[str]): A list of column names that represent the targets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
            - X_train_processed (pd.DataFrame): Preprocessed training features.
            - y_train (pd.DataFrame): Training target columns.
            - X_test_processed (pd.DataFrame): Preprocessed test features.
            - test_ids (pd.Series): The 'id' column from the test set.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """

    # Store 'id' column for potential submission file generation
    test_ids = test_df['id']

    # Define feature columns (all columns except 'id' and target columns)
    feature_columns = [col for col in train_df.columns if col not in target_columns + ['id']]

    # Separate features and targets for training data
    X_train = train_df[feature_columns]
    y_train = train_df[target_columns]

    # Select feature columns for test data (assuming test.csv does not contain target columns)
    X_test = test_df[feature_columns]

    # All identified features are numerical based on the dataset metadata.
    # Binary columns like 'TypeOfSteel_A300' and 'TypeOfSteel_A400' are treated as numerical
    # and will be scaled along with other numerical features.
    numerical_features = feature_columns

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training features and transform both training and test features
    X_train_scaled = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled = scaler.transform(X_test[numerical_features])

    # Convert scaled arrays back to DataFrames, preserving column names and index
    X_train_processed = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

    print("\nData preprocessing completed.")
    print(f"Shape of preprocessed training features (X_train_processed): {X_train_processed.shape}")
    print(f"Shape of training targets (y_train): {y_train.shape}")
    print(f"Shape of preprocessed test features (X_test_processed): {X_test_processed.shape}")

    return X_train_processed, y_train, X_test_processed, test_ids, scaler

def train_and_evaluate_model(
    X_train_processed: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test_processed: pd.DataFrame, 
    test_ids: pd.Series, 
    target_columns: list[str]
) -> dict:
    """
    Trains a LightGBM model for each target column, evaluates performance,
    and persists the trained models.

    Args:
        X_train_processed (pd.DataFrame): Preprocessed training features.
        y_train (pd.DataFrame): Training target columns.
        X_test_processed (pd.DataFrame): Preprocessed test features.
        test_ids (pd.Series): The 'id' column from the test set.
        target_columns (list[str]): A list of column names that represent the targets.

    Returns:
        dict: A dictionary containing the trained models, one for each target.
    """
    print("\n--- Model Training and Evaluation ---")

    # Perform an 80/20 stratified split into training and validation sets
    # For multi-label classification, stratify on one of the target columns.
    # 'Other_Faults' is chosen as it's the least imbalanced among the targets.
    # Ensure y_train['Other_Faults'] is not all zeros or all ones for stratification to work.
    # If it is, stratification will fail. Based on EDA, it's not all zeros/ones.
    try:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train['Other_Faults']
        )
    except ValueError as e:
        print(f"Warning: Stratification failed for 'Other_Faults'. Falling back to non-stratified split. Error: {e}")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_processed, y_train, test_size=0.2, random_state=42
        )

    print(f"Training split shapes: X_train_split={X_train_split.shape}, y_train_split={y_train_split.shape}")
    print(f"Validation split shapes: X_val_split={X_val_split.shape}, y_val_split={y_val_split.shape}")

    trained_models = {}
    val_preds_list = []
    val_probas_list = []

    for target_col in target_columns:
        print(f"\nTraining model for target: {target_col}")

        # Initialize LGBMClassifier for binary classification
        model = LGBMClassifier(objective='binary', metric='logloss', random_state=42, n_estimators=1000)

        # Fit the model with early stopping
        # Corrected: Use lgb.early_stopping instead of LGBMClassifier.early_stopping
        model.fit(X_train_split, y_train_split[target_col],
                  eval_set=[(X_val_split, y_val_split[target_col])],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

        trained_models[target_col] = model

        # Make predictions and probabilities on the validation set
        val_pred = model.predict(X_val_split)
        val_proba = model.predict_proba(X_val_split)[:, 1] # Probability of the positive class

        val_preds_list.append(val_pred)
        val_probas_list.append(val_proba)

    # Convert lists of predictions/probabilities to DataFrames
    y_val_pred = pd.DataFrame(val_preds_list).T
    y_val_pred.columns = target_columns
    y_val_proba = pd.DataFrame(val_probas_list).T
    y_val_proba.columns = target_columns

    # --- Evaluate Metrics ---
    print("\n--- Evaluation Metrics on Validation Set ---")

    overall_accuracy = []
    overall_f1_macro = []
    overall_f1_micro = []
    overall_logloss = []

    for i, target_col in enumerate(target_columns):
        y_true_col = y_val_split[target_col]
        y_pred_col = y_val_pred[target_col]
        y_proba_col = y_val_proba[target_col]

        acc = accuracy_score(y_true_col, y_pred_col)
        # Handle cases where F1 score might be undefined due to no positive samples
        try:
            f1_macro = f1_score(y_true_col, y_pred_col, average='macro', zero_division=0)
        except ValueError: # Raised if there are no positive samples in y_true or y_pred
            f1_macro = 0.0
        try:
            f1_micro = f1_score(y_true_col, y_pred_col, average='micro', zero_division=0)
        except ValueError:
            f1_micro = 0.0

        ll = log_loss(y_true_col, y_proba_col)

        overall_accuracy.append(acc)
        overall_f1_macro.append(f1_macro)
        overall_f1_micro.append(f1_micro)
        overall_logloss.append(ll)

        print(f"Metrics for {target_col}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (Macro): {f1_macro:.4f}")
        print(f"  F1 (Micro): {f1_micro:.4f}")
        print(f"  LogLoss: {ll:.4f}")

    print("\n--- Average Metrics Across All Targets ---")
    print(f"Average Accuracy: {sum(overall_accuracy) / len(overall_accuracy):.4f}")
    print(f"Average F1 (Macro): {sum(overall_f1_macro) / len(overall_f1_macro):.4f}")
    print(f"Average F1 (Micro): {sum(overall_f1_micro) / len(overall_f1_micro):.4f}")
    print(f"Average LogLoss: {sum(overall_logloss) / len(overall_logloss):.4f}")

    # --- Persist the trained models ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(trained_models, MODEL_PATH)
    print(f"\nTrained models saved to {MODEL_PATH}")

    return trained_models

def main():
    """
    Main function to orchestrate the data loading, preprocessing, training, and evaluation pipeline.
    """

    # 1. Load the dataset
    train_df, test_df = load_data(BASE_PATH)

    # 2. Perform preprocessing
    X_train_processed, y_train, X_test_processed, test_ids, scaler = preprocess_data(
        train_df.copy(), test_df.copy(), TARGET_COLUMNS
    )

    # 3. Train and evaluate the model
    trained_model = train_and_evaluate_model(
        X_train_processed, y_train, X_test_processed, test_ids, TARGET_COLUMNS
    )

    # Return the trained model instance for potential further use (e.g., prediction on test_df)
    return trained_model

if __name__ == "__main__":
    # Execute the main pipeline when the script is run
    trained_model_instance = main()

    # Display a confirmation of the trained model
    print("\n--- Pipeline Execution Complete ---")
    print(f"Trained model instance (dictionary of LGBMClassifiers): {trained_model_instance}")
    print(f"Keys in trained_model_instance: {list(trained_model_instance.keys())}")

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib # For loading the trained models

# --- File Path Constants ---
# Default paths mirroring the standard Kaggle notebook directory layout
KAGGLE_BASE_PATH = "./"

# Fallback for local development or different environments
# Adjust this path if your local dataset directory structure is different
LOCAL_BASE_PATH = "./" 

# Model persistence path (from previous training script)
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "steel_plate_defect_prediction_model.pkl")

# Submission file path
SUBMISSION_DIR = "./outputs"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Define target columns based on the dataset metadata
TARGET_COLUMNS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

def load_data(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test datasets from the specified base path.
    This function is re-used from the preprocessing script to ensure consistency.

    Args:
        base_path (str): The base directory where the dataset files are located.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"Successfully loaded train.csv from: {train_path}")
        print(f"Successfully loaded test.csv from: {test_path}")

        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check the BASE_PATH and file names.")
        raise

def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
    """
    Performs preprocessing steps on the dataset.
    This function is re-used from the preprocessing script to ensure consistency.
    The `train_df` is required to fit the StandardScaler, even when only processing `test_df` for prediction.

    Args:
        train_df (pd.DataFrame): The raw training DataFrame (used for scaler fitting).
        test_df (pd.DataFrame): The raw test DataFrame.
        target_columns (list[str]): A list of column names that represent the targets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler]:
            - X_train_processed (pd.DataFrame): Preprocessed training features (not directly used here, but returned).
            - y_train (pd.DataFrame): Training target columns (not directly used here, but returned).
            - X_test_processed (pd.DataFrame): Preprocessed test features.
            - test_ids (pd.Series): The 'id' column from the test set.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """

    # Store 'id' column for potential submission file generation
    test_ids = test_df['id']

    # Define feature columns (all columns except 'id' and target columns)
    feature_columns = [col for col in train_df.columns if col not in target_columns + ['id']]

    # Separate features and targets for training data
    X_train = train_df[feature_columns]
    y_train = train_df[target_columns] # y_train is not strictly needed for test preprocessing, but returned by the function signature

    # Select feature columns for test data (assuming test.csv does not contain target columns)
    X_test = test_df[feature_columns]

    # All identified features are numerical based on the dataset metadata.
    numerical_features = feature_columns

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training features and transform both training and test features
    X_train_scaled = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled = scaler.transform(X_test[numerical_features])

    # Convert scaled arrays back to DataFrames, preserving column names and index
    X_train_processed = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

    print("\nData preprocessing completed for prediction.")
    print(f"Shape of preprocessed test features (X_test_processed): {X_test_processed.shape}")

    return X_train_processed, y_train, X_test_processed, test_ids, scaler

def predict_and_submit():
    """
    Main function to orchestrate the prediction and submission file generation pipeline.
    """

    # Determine the actual base path based on existence
    if os.path.exists(KAGGLE_BASE_PATH):
        BASE_PATH = KAGGLE_BASE_PATH
    elif os.path.exists(LOCAL_BASE_PATH):
        BASE_PATH = LOCAL_BASE_PATH
    else:
        raise FileNotFoundError(f"Dataset base path not found at {KAGGLE_BASE_PATH} or {LOCAL_BASE_PATH}")

    print(f"Using BASE_PATH: {BASE_PATH}")

    # 1. Load the dataset
    # train_df is loaded because it's needed by preprocess_data to fit the scaler.
    train_df, test_df = load_data(BASE_PATH)

    # 2. Preprocess data to get X_test_processed and test_ids
    # The previous error was in Stage 2 related to LightGBM's early stopping callback.
    # This Stage 3 code assumes Stage 2 has been successfully executed and models are saved.
    # X_test_processed is generated in-memory here, consistent with Stage 1's output.
    _, _, X_test_processed, test_ids, _ = preprocess_data(
        train_df.copy(), test_df.copy(), TARGET_COLUMNS
    )

    # 3. Load the trained models
    print(f"\nAttempting to load trained models from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained models not found at {MODEL_PATH}. "
                                "Please ensure the training script has been run successfully to save the models.")

    trained_models = joblib.load(MODEL_PATH)
    print("Trained models loaded successfully.")
    print(f"Models loaded for targets: {list(trained_models.keys())}")

    # 4. Generate predictions (probabilities for multi-label classification)
    print("\nGenerating predictions on the test set...")
    test_predictions_proba = {}
    for target_col in TARGET_COLUMNS:
        if target_col in trained_models:
            model = trained_models[target_col]
            # Predict probabilities for the positive class (class 1)
            test_predictions_proba[target_col] = model.predict_proba(X_test_processed)[:, 1]
            print(f"  Predictions generated for '{target_col}'.")
        else:
            print(f"  Warning: No model found for target '{target_col}'. Skipping and setting probabilities to 0.0.")
            # This fallback ensures the submission file can still be created even if a model is missing.
            test_predictions_proba[target_col] = [0.0] * len(X_test_processed) 

    # 5. Build submission DataFrame
    print("\nBuilding submission file...")
    submission_df = pd.DataFrame({'id': test_ids})
    for target_col in TARGET_COLUMNS:
        submission_df[target_col] = test_predictions_proba[target_col]

    # 6. Save the submission file
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSubmission file generated successfully at: {SUBMISSION_PATH}")
    print(f"Submission file head:\n{submission_df.head()}")
    print(f"Submission file shape: {submission_df.shape}")

if __name__ == "__main__":
    predict_and_submit()

if __name__ == "__main__":
    main()
