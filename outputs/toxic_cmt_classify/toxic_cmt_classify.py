import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from pathlib import Path
import joblib
import json
import numpy as np
import warnings

# Suppress specific warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# --- File Path Constants ---
# Determine the project root dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent

# Base path for datasets, with fallback for different execution environments
BASE_PATH_OPTION1 = (ROOT_DIR / 'input/Datasets/datasets/toxic_cmt_classify').resolve()
BASE_PATH_OPTION2 = Path('input/Datasets/datasets/toxic_cmt_classify').resolve()

if BASE_PATH_OPTION1.exists():
    BASE_PATH = BASE_PATH_OPTION1
else:
    BASE_PATH = BASE_PATH_OPTION2

TRAIN_FILE = BASE_PATH / "jigsaw-toxic-comment-train.csv"
TEST_FILE = BASE_PATH / "test.csv"
SAMPLE_SUBMISSION_FILE = BASE_PATH / "sample_submission.csv"

# Define output paths
OUTPUT_DIR = Path("./outputs")
MODELS_DIR = Path("./models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "toxic_cmt_classify_model.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

def main():
    """
    Orchestrates the data loading, cleaning, preprocessing, model training,
    and evaluation steps for the toxic comment classification dataset.
    This function now encapsulates the full pipeline from Stage 1 and Stage 2
    to ensure all necessary components (like the vectorizer) are available
    for prediction if the model is loaded.
    """
    print(f"Resolved BASE_PATH: {BASE_PATH}")

    # --- Load Data ---
    print("Loading data...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure the dataset files are in the correct directory.")
        return None, None, None, None, None # Return None to indicate failure

    # --- Dataset Overview (from EDA and Metadata) ---
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    text_column = 'comment_text'
    id_column = 'id'

    # --- Preprocessing ---

    # 1. Handle Missing Values (EDA shows none, but good practice)
    print("Checking for missing values...")
    if train_df[text_column].isnull().any():
        print(f"Warning: Missing values found in '{text_column}' in train_df. Filling with empty string.")
        train_df[text_column].fillna('', inplace=True)
    if test_df[text_column].isnull().any():
        print(f"Warning: Missing values found in '{text_column}' in test_df. Filling with empty string.")
        test_df[text_column].fillna('', inplace=True)
    else:
        print("No missing values found in critical columns based on EDA.")

    # 2. Text Feature Engineering (TF-IDF)
    print(f"Applying TF-IDF to '{text_column}'...")
    tfidf_vectorizer = TfidfVectorizer(max_features=20000, min_df=3, max_df=0.85, ngram_range=(1, 2))

    X_train_text = tfidf_vectorizer.fit_transform(train_df[text_column])
    X_test_text = tfidf_vectorizer.transform(test_df[text_column])

    print(f"TF-IDF transformation complete. Train shape: {X_train_text.shape}, Test shape: {X_test_text.shape}")

    # 3. Prepare Target Variables
    y_train_labels = train_df[target_columns]
    print("Target variables prepared.")

    # --- Data Splitting (80/20 stratified split) ---
    print("Splitting data into training and validation sets (80/20 stratified)...")
    # Stratify on 'toxic' as it's the main target and other labels are highly imbalanced.
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_text, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels['toxic']
    )

    print(f"Training split shapes: X_train_split={X_train_split.shape}, y_train_split={y_train_split.shape}")
    print(f"Validation split shapes: X_val_split={X_val_split.shape}, y_val_split={y_val_split.shape}")

    # --- Model Building and Training ---
    print("\n--- Model Training ---")
    model = OneVsRestClassifier(LogisticRegression(solver='saga', C=0.5, n_jobs=-1, random_state=42, max_iter=1000))

    print("Training model (OneVsRestClassifier with Logistic Regression)...")
    model.fit(X_train_split, y_train_split)
    print("Model training complete.")

    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    metrics = {}
    overall_roc_auc_scores = []
    overall_accuracy_scores = []
    overall_f1_scores = []
    overall_log_loss_scores = []

    y_val_pred_proba = model.predict_proba(X_val_split)
    y_val_pred_labels = (y_val_pred_proba > 0.5).astype(int)

    for i, label in enumerate(target_columns):
        label_metrics = {}
        unique_true_classes = np.unique(y_val_split[label])

        # ROC AUC and Log Loss require at least two classes in true labels
        if len(unique_true_classes) > 1:
            try:
                auc = roc_auc_score(y_val_split[label], y_val_pred_proba[:, i])
                label_metrics['roc_auc'] = auc
                overall_roc_auc_scores.append(auc)
            except ValueError as e:
                print(f"  Warning: Could not calculate ROC AUC for '{label}': {e}")
                label_metrics['roc_auc'] = None

            try:
                ll = log_loss(y_val_split[label], y_val_pred_proba[:, i])
                label_metrics['log_loss'] = ll
                overall_log_loss_scores.append(ll)
            except ValueError as e:
                print(f"  Warning: Could not calculate Log Loss for '{label}': {e}")
                label_metrics['log_loss'] = None
        else:
            print(f"  Warning: Only one class present in true labels for '{label}'. Skipping ROC AUC and Log Loss.")
            label_metrics['roc_auc'] = None
            label_metrics['log_loss'] = None

        try:
            acc = accuracy_score(y_val_split[label], y_val_pred_labels[:, i])
            label_metrics['accuracy'] = acc
            overall_accuracy_scores.append(acc)
        except ValueError as e:
            print(f"  Warning: Could not calculate Accuracy for '{label}': {e}")
            label_metrics['accuracy'] = None

        try:
            f1 = f1_score(y_val_split[label], y_val_pred_labels[:, i], average='binary', zero_division=0)
            label_metrics['f1_score'] = f1
            overall_f1_scores.append(f1)
        except ValueError as e:
            print(f"  Warning: Could not calculate F1 Score for '{label}': {e}")
            label_metrics['f1_score'] = None

        print(f"  Metrics for '{label}':")
        print(f"    ROC AUC: {label_metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"    Accuracy: {label_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"    F1 Score: {label_metrics.get('f1_score', 'N/A'):.4f}")
        print(f"    Log Loss: {label_metrics.get('log_loss', 'N/A'):.4f}")

        metrics[label] = label_metrics

    # Calculate overall metrics by averaging across labels
    filtered_roc_auc = [score for score in overall_roc_auc_scores if score is not None]
    filtered_accuracy = [score for score in overall_accuracy_scores if score is not None]
    filtered_f1 = [score for score in overall_f1_scores if score is not None]
    filtered_log_loss = [score for score in overall_log_loss_scores if score is not None]

    metrics['overall_average'] = {}
    if filtered_roc_auc:
        metrics['overall_average']['roc_auc'] = np.mean(filtered_roc_auc)
        print(f"\nOverall Average ROC AUC: {metrics['overall_average']['roc_auc']:.4f}")
    else:
        metrics['overall_average']['roc_auc'] = "N/A"
        print("\nOverall Average ROC AUC: N/A (could not calculate for any label)")

    if filtered_accuracy:
        metrics['overall_average']['accuracy'] = np.mean(filtered_accuracy)
        print(f"Overall Average Accuracy: {metrics['overall_average']['accuracy']:.4f}")
    else:
        metrics['overall_average']['accuracy'] = "N/A"
        print("Overall Average Accuracy: N/A (could not calculate for any label)")

    if filtered_f1:
        metrics['overall_average']['f1_score'] = np.mean(filtered_f1)
        print(f"Overall Average F1 Score: {metrics['overall_average']['f1_score']:.4f}")
    else:
        metrics['overall_average']['f1_score'] = "N/A"
        print("Overall Average F1 Score: N/A (could not calculate for any label)")

    if filtered_log_loss:
        metrics['overall_average']['log_loss'] = np.mean(filtered_log_loss)
        print(f"Overall Average Log Loss: {metrics['overall_average']['log_loss']:.4f}")
    else:
        metrics['overall_average']['log_loss'] = "N/A"
        print("Overall Average Log Loss: N/A (could not calculate for any label)")

    # Persist metrics to JSON file
    print(f"Saving metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")

    # --- Persist Trained Model and TF-IDF Vectorizer ---
    print(f"Saving trained model and TF-IDF vectorizer to {MODEL_PATH}...")
    joblib.dump({'model': model, 'vectorizer': tfidf_vectorizer}, MODEL_PATH)
    print("Model and TF-IDF vectorizer saved.")

    # Return necessary components for prediction
    return model, X_test_text, test_df[id_column], target_columns, tfidf_vectorizer # Also return vectorizer

def generate_predictions(trained_model=None, X_test_processed=None, test_ids=None, target_columns=None, tfidf_vectorizer=None):
    """
    Generates predictions using the trained model and saves them to a submission file.
    """
    print("\n--- Generating Submission File ---")

    # 1. Ensure trained_model and vectorizer are available. If None, load from disk.
    if trained_model is None or tfidf_vectorizer is None:
        print(f"No trained model or vectorizer instance provided. Attempting to load from {MODEL_PATH}...")
        try:
            loaded_artifacts = joblib.load(MODEL_PATH)
            trained_model = loaded_artifacts['model']
            tfidf_vectorizer = loaded_artifacts['vectorizer']
            print("Model and vectorizer loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}. Cannot generate predictions.")
            return
        except Exception as e:
            print(f"Error loading model: {e}. Cannot generate predictions.")
            return

        # If X_test_processed or test_ids or target_columns are also None,
        # it means we are running prediction independently and need to re-process the test data.
        if X_test_processed is None or test_ids is None or target_columns is None:
            print("Re-loading test data and re-processing for prediction...")
            try:
                test_df = pd.read_csv(TEST_FILE)
                text_column = 'comment_text'
                id_column = 'id'
                target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # Re-define
                test_ids = test_df[id_column]

                if test_df[text_column].isnull().any():
                    test_df[text_column].fillna('', inplace=True)

                X_test_processed = tfidf_vectorizer.transform(test_df[text_column])
                print(f"Test data re-processed. Shape: {X_test_processed.shape}")

            except FileNotFoundError as e:
                print(f"Error loading test data: {e}. Cannot generate predictions.")
                return
            except Exception as e:
                print(f"Error re-processing test data: {e}. Cannot generate predictions.")
                return

    # 2. Generate predictions (predict_proba for classification)
    print("Generating test predictions...")
    test_predictions_proba = trained_model.predict_proba(X_test_processed)

    # 3. Build submission_df following the sample submission format
    # The problem asks for 'toxic' probability.
    toxic_col_idx = target_columns.index('toxic')
    submission_df = pd.DataFrame({
        'id': test_ids,
        'toxic': test_predictions_proba[:, toxic_col_idx]
    })

    # 4. Save the submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    # 5. Print confirmation message
    print(f"Submission file saved to {SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == "__main__":
    # Run the main training and evaluation pipeline
    trained_model_instance, X_test_processed_data, test_ids_data, target_cols, tfidf_vectorizer_instance = main()

    # Generate predictions using the returned model and processed data
    # This ensures that if main() successfully trained a model, we use that instance
    # and the already processed test data, avoiding re-loading/re-processing.
    if trained_model_instance is not None:
        generate_predictions(trained_model=trained_model_instance,
                             X_test_processed=X_test_processed_data,
                             test_ids=test_ids_data,
                             target_columns=target_cols,
                             tfidf_vectorizer=tfidf_vectorizer_instance)
    else:
        print("Model training failed. Attempting to generate predictions by loading saved model (if available).")
        # If main() failed, try to generate predictions by loading from disk
        generate_predictions()

    print("\nScript execution finished successfully.")