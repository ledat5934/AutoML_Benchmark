import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
import os
import json
import joblib

# Determine the project root
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:  # __file__ is not defined inside Kaggle/Jupyter
    ROOT_DIR = Path.cwd()

# Define base path with fallback
BASE_PATH_CANDIDATE_1 = (ROOT_DIR / 'input/Datasets/datasets/dog_breed_classification').resolve()
BASE_PATH_CANDIDATE_2 = Path('input/Datasets/datasets/dog_breed_classification').resolve()

if BASE_PATH_CANDIDATE_1.exists():
    BASE_PATH = BASE_PATH_CANDIDATE_1
else:
    BASE_PATH = BASE_PATH_CANDIDATE_2

print(f"Resolved BASE_PATH: {BASE_PATH}")

# Define file paths
TRAIN_LABELS_PATH = BASE_PATH / 'labels.csv'
SAMPLE_SUBMISSION_PATH = BASE_PATH / 'sample_submission.csv'
TRAIN_IMAGES_DIR = BASE_PATH / 'train'
TEST_IMAGES_DIR = BASE_PATH / 'test'

# Output paths
OUTPUT_DIR = ROOT_DIR / 'outputs'
MODELS_DIR = ROOT_DIR / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "dog_breed_classification_model.keras" # Keras models are saved with .keras extension
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib" # Path to save the label encoder
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# Image preprocessing constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def load_image(image_path, label=None):
    """Loads and preprocesses an image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    if label is not None:
        return img, label
    return img

def create_image_dataset(image_paths, labels=None, shuffle=False, augment=False):
    """Creates a tf.data.Dataset for images."""
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # No explicit augmentation for now, but can be added here

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(num_classes):
    """Builds a fine-tuned EfficientNetB0 model."""
    print("Building EfficientNetB0 model...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Freeze the base model layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Model summary:")
    model.summary()
    return model

def main():
    """
    Main function to orchestrate data loading, preprocessing, model training, evaluation, and prediction.
    """
    print("--- Stage 1: Data Loading and Preprocessing ---")

    # 1. Load Labels
    print(f"Loading training labels from: {TRAIN_LABELS_PATH}")
    try:
        labels_df = pd.read_csv(TRAIN_LABELS_PATH)
        print(f"Labels DataFrame head:\n{labels_df.head()}")
        print(f"Labels DataFrame shape: {labels_df.shape}")
    except FileNotFoundError:
        print(f"Error: {TRAIN_LABELS_PATH} not found. Please check the path.")
        return

    # 2. Encode Breed Labels
    print("Encoding breed labels...")
    label_encoder = LabelEncoder()
    labels_df['breed_encoded'] = label_encoder.fit_transform(labels_df['breed'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of unique breeds (classes): {num_classes}")
    print(f"Encoded labels head:\n{labels_df.head()}")

    # Save the label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    # 3. Prepare Image Paths
    print("Preparing image paths...")
    labels_df['image_path'] = labels_df['id'].apply(lambda x: str(TRAIN_IMAGES_DIR / f"{x}.jpg"))

    # Verify all image paths exist
    missing_images = [path for path in labels_df['image_path'] if not Path(path).exists()]
    if missing_images:
        print(f"Warning: {len(missing_images)} training images not found. Example: {missing_images[0]}")
        # Filter out rows with missing images if necessary, or handle as an error
        labels_df = labels_df[labels_df['image_path'].apply(lambda x: Path(x).exists())]
        print(f"Filtered labels DataFrame shape after removing missing images: {labels_df.shape}")
    else:
        print("All training image paths verified.")

    # 4. Split Data into Training and Validation Sets
    print("Splitting data into training and validation sets (80/20 stratified split)...")
    train_df, val_df = train_test_split(
        labels_df,
        test_size=0.2,
        stratify=labels_df['breed_encoded'],
        random_state=42
    )
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # 5. Create TensorFlow Datasets for Training and Validation
    print("Creating TensorFlow Datasets for training and validation...")
    train_image_paths = train_df['image_path'].values
    train_labels_encoded = train_df['breed_encoded'].values
    val_image_paths = val_df['image_path'].values
    val_labels_encoded = val_df['breed_encoded'].values

    train_dataset = create_image_dataset(train_image_paths, train_labels_encoded, shuffle=True)
    val_dataset = create_image_dataset(val_image_paths, val_labels_encoded)

    print("Training dataset created.")
    print("Validation dataset created.")

    # 6. Prepare Test Image Paths
    print("Preparing test image paths...")
    test_image_ids = [Path(f).stem for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
    test_image_paths = [str(TEST_IMAGES_DIR / f"{img_id}.jpg") for img_id in test_image_ids]

    # Verify all test image paths exist
    missing_test_images = [path for path in test_image_paths if not Path(path).exists()]
    if missing_test_images:
        print(f"Warning: {len(missing_test_images)} test images not found. Example: {missing_test_images[0]}")
        # Filter out missing test images
        test_image_paths = [path for path in test_image_paths if Path(path).exists()]
        print(f"Filtered test image count: {len(test_image_paths)}")
    else:
        print("All test image paths verified.")

    # 7. Create TensorFlow Dataset for Test Images (without labels)
    print("Creating TensorFlow Dataset for test images...")
    test_dataset = create_image_dataset(test_image_paths)
    print("Test dataset created.")

    print("--- Data Preprocessing Complete ---")

    print("\n--- Stage 2: Model Training and Evaluation ---")

    # Build the model
    model = build_model(num_classes)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_dataset,
        epochs=100, # Set a high number, early stopping will stop it
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating the model on the validation set...")
    val_predictions_proba = model.predict(val_dataset)
    val_predictions = np.argmax(val_predictions_proba, axis=1)

    metrics = {}
    metrics['overall'] = {
        'accuracy': accuracy_score(val_labels_encoded, val_predictions),
        'f1_score_weighted': f1_score(val_labels_encoded, val_predictions, average='weighted'),
        'log_loss': log_loss(val_labels_encoded, val_predictions_proba)
    }

    try:
        metrics['overall']['roc_auc_ovr'] = roc_auc_score(val_labels_encoded, val_predictions_proba, multi_class='ovr')
    except ValueError as e:
        print(f"Could not calculate ROC AUC (OvR): {e}. This might happen if there's only one class in a fold or other issues.")
        metrics['overall']['roc_auc_ovr'] = None

    print("\n--- Evaluation Metrics ---")
    for metric_name, value in metrics['overall'].items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")

    # Persist metrics to JSON
    print(f"Saving metrics to {METRICS_PATH}")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")

    # Persist the trained model
    print(f"Saving trained model to {MODEL_PATH}")
    model.save(MODEL_PATH) # Saves in Keras native format
    print("Model saved.")

    print("--- Model Training and Evaluation Complete ---")

    print("\n--- Stage 3: Prediction and Submission ---")

    # Generate predictions (probabilities for multi-class classification)
    print("Generating predictions on the test set...")
    test_predictions_proba = model.predict(test_dataset)
    print(f"Predictions shape: {test_predictions_proba.shape}")

    # Load sample submission to match format
    print(f"Loading sample submission from: {SAMPLE_SUBMISSION_PATH}")
    try:
        sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        print(f"Sample submission head:\n{sample_submission_df.head()}")
    except FileNotFoundError:
        print(f"Error: {SAMPLE_SUBMISSION_PATH} not found. Cannot create submission file.")
        return

    # Create submission DataFrame
    submission_df = pd.DataFrame({'id': test_image_ids})

    # Get breed names from label encoder
    breed_names = label_encoder.classes_

    # Add probability columns for each breed
    for i, breed in enumerate(breed_names):
        submission_df[breed] = test_predictions_proba[:, i]

    # Ensure the order of columns matches sample submission (excluding 'id')
    # This is crucial for Kaggle submissions
    submission_columns = ['id'] + list(sample_submission_df.columns.drop('id'))
    submission_df = submission_df[submission_columns]

    # Save the submission file
    print(f"Saving submission file to {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file generated successfully at {SUBMISSION_PATH}")
    print(f"Submission DataFrame head:\n{submission_df.head()}")
    print(f"Submission DataFrame shape: {submission_df.shape}")

    print("--- Prediction and Submission Complete ---")


if __name__ == "__main__":
    main()