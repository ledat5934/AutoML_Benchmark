import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import json
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, cohen_kappa_score
import joblib

# Global constants (or could be passed as arguments to main)
IMG_SIZE = (224, 224) # Standard input size for many pre-trained models

def load_and_preprocess_image(image_path):
    """Loads, resizes, and normalizes an image."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0 # Normalize to [0,1]
    return img

def get_image_paths(pet_id, base_dir, photo_amt):
    """Generates potential image paths for a given PetID."""
    paths = []
    for i in range(photo_amt):
        img_path = base_dir / f"{pet_id}-{i+1}.jpg"
        if img_path.exists():
            paths.append(str(img_path))
    return paths

def extract_image_features(df, base_image_dir, image_feature_extractor):
    """Extracts image features for each pet."""
    if image_feature_extractor is None:
        # Return a DataFrame with PetID as index and zero-filled columns
        # to ensure consistent feature set even if image features are skipped.
        # The feature vector size for EfficientNetB0 is 1280.
        dummy_features = np.zeros((len(df), 1280))
        feature_df = pd.DataFrame(dummy_features, index=df['PetID'])
        feature_df.columns = [f'img_feat_{i}' for i in range(feature_df.shape[1])]
        return feature_df

    all_pet_features = []
    pet_ids_with_features = []

    for index, row in df.iterrows():
        pet_id = row['PetID']
        photo_amt = int(row['PhotoAmt']) # Ensure PhotoAmt is integer
        image_paths = get_image_paths(pet_id, base_image_dir, photo_amt)

        if image_paths:
            images = []
            for path in image_paths:
                try:
                    images.append(load_and_preprocess_image(path))
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue

            if images:
                images_tensor = tf.stack(images) # Stack into a batch
                # Extract features
                features = image_feature_extractor(images_tensor)
                # Aggregate features (e.g., mean pooling across images for a single pet)
                aggregated_features = tf.reduce_mean(features, axis=0).numpy()
                all_pet_features.append(aggregated_features)
                pet_ids_with_features.append(pet_id)
            else:
                # If no images could be loaded, append zeros
                all_pet_features.append(np.zeros(1280))
                pet_ids_with_features.append(pet_id)
        else:
            # If no images found, append zeros
            all_pet_features.append(np.zeros(1280))
            pet_ids_with_features.append(pet_id)

    # Create a DataFrame from extracted features
    feature_df = pd.DataFrame(all_pet_features, index=pet_ids_with_features)
    feature_df.columns = [f'img_feat_{i}' for i in range(feature_df.shape[1])]
    return feature_df

def get_base_paths():
    """Determines the root and base paths for the dataset."""
    try:
        ROOT_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        ROOT_DIR = Path.cwd()

    BASE_PATH_CANDIDATE = (ROOT_DIR / 'input/Datasets/datasets/pet_finder').resolve()
    if BASE_PATH_CANDIDATE.exists():
        BASE_PATH = BASE_PATH_CANDIDATE
    else:
        BASE_PATH = Path('input/Datasets/datasets/pet_finder').resolve()
    print(f"Resolved BASE_PATH: {BASE_PATH}")
    return ROOT_DIR, BASE_PATH

def load_data(base_path):
    """Loads the main datasets and label files."""
    TRAIN_CSV_PATH = base_path / 'train.csv'
    TEST_CSV_PATH = base_path / 'test/test.csv'
    BREED_LABELS_PATH = base_path / 'BreedLabels.csv'
    COLOR_LABELS_PATH = base_path / 'ColorLabels.csv'
    STATE_LABELS_PATH = base_path / 'StateLabels.csv'
    SAMPLE_SUBMISSION_PATH = base_path / 'test/sample_submission.csv'

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    breed_labels = pd.read_csv(BREED_LABELS_PATH)
    color_labels = pd.read_csv(COLOR_LABELS_PATH)
    state_labels = pd.read_csv(STATE_LABELS_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    return train_df, test_df, breed_labels, color_labels, state_labels, sample_submission_df

def get_dataset_metadata():
    """Returns the hardcoded dataset metadata JSON."""
    return {
      "dataset_info": {
        "name": "pet_finder",
        "base_path": "input/Datasets/datasets/pet_finder",
        "description_file": "description.txt",
        "files": [
          {
            "path": "BreedLabels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "breed_labels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "ColorLabels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "color_labels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "StateLabels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "state_labels.csv",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "train.csv",
            "role": "train",
            "type": "tabular"
          },
          {
            "path": "test/sample_submission.csv",
            "role": "sample",
            "type": "tabular"
          },
          {
            "path": "test/test.csv",
            "role": "test",
            "type": "tabular"
          },
          {
            "path": "pet_finder.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/002230dea-1.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/002230dea-2.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/002230dea-3.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/002230dea-4.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/0063f83c9-1.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/0063f83c9-2.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/0063f83c9-3.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/0063f83c9-4.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "test_metadata/0073c33d0-1.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "00bfa5da9-1.json",
            "role": "data",
            "type": "tabular"
          },
          {
            "path": "<omitted>",
            "role": "bulk_files_summary",
            "type": "summary",
            "omitted_count": 72749
          }
        ]
      },
      "profiling_summary": {
        "time_index_analysis": "None",
        "table": {
          "n": 11994,
          "n_var": 24,
          "memory_size": 2302976,
          "record_size": 192.010672002668,
          "n_cells_missing": 1022,
          "p_cells_missing": 0.0035503863042632426,
          "size_optimized": True,
          "optimization_level": "aggressive",
          "optimization_note": "All value lists removed - only counts and basic statistics retained",
          "removed_sections": 129,
          "optimization_strategy": "Minimal JSON for maximum compatibility with LLM token limits"
        },
        "variables": {
          "Type": {
            "n_distinct": 2,
            "p_distinct": 0.00016675004168751042,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.455060863765216,
            "std": 0.49799713968618003,
            "variance": 0.24800115113561672,
            "min": 1,
            "max": 2,
            "kurtosis": -1.9677444352623312,
            "skewness": 0.180509595359183,
            "sum": 17452,
            "mad": 0.0,
            "range": 1,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 1.0,
            "75%": 2.0,
            "95%": 2.0,
            "iqr": 1.0,
            "cv": 0.34225175873229674,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Name": {
            "n_distinct": 7433,
            "p_distinct": 0.6768348206155527,
            "is_unique": False,
            "n_unique": 6422,
            "p_unique": 0.5847750865051903,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 1012,
            "n": 11994,
            "p_missing": 0.08437552109388027,
            "count": 10982,
            "memory_size": 96080,
            "max_length": 47,
            "mean_length": 9.545255873247132,
            "median_length": 42,
            "min_length": 1,
            "n_characters_distinct": 167,
            "n_characters": 104826,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "Age": {
            "n_distinct": 103,
            "p_distinct": 0.008587627146906788,
            "is_unique": False,
            "n_unique": 26,
            "p_unique": 0.0021677505419376354,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 139,
            "mean": 10.5200100050025,
            "std": 18.333786187461325,
            "variance": 336.1277159675477,
            "min": 0,
            "max": 255,
            "kurtosis": 22.350465263684598,
            "skewness": 3.8602028718592316,
            "sum": 126177,
            "mad": 2.0,
            "range": 255,
            "5%": 1.0,
            "25%": 2.0,
            "50%": 3.0,
            "75%": 12.0,
            "95%": 48.0,
            "iqr": 10.0,
            "cv": 1.7427536835747492,
            "p_zeros": 0.011589127897281974,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Breed1": {
            "n_distinct": 166,
            "p_distinct": 0.013840253460063364,
            "is_unique": False,
            "n_unique": 38,
            "p_unique": 0.003168250792062698,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 3,
            "mean": 265.1770885442721,
            "std": 60.087706445934444,
            "variance": 3610.5324659327916,
            "min": 0,
            "max": 307,
            "kurtosis": 4.755229972607767,
            "skewness": -2.20387682115463,
            "sum": 3180534,
            "mad": 41.0,
            "range": 307,
            "5%": 109.0,
            "25%": 265.0,
            "50%": 266.0,
            "75%": 307.0,
            "95%": 307.0,
            "iqr": 42.0,
            "cv": 0.2265946382313592,
            "p_zeros": 0.0002501250625312656,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Breed2": {
            "n_distinct": 127,
            "p_distinct": 0.010588627647156911,
            "is_unique": False,
            "n_unique": 41,
            "p_unique": 0.0034183758545939637,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 8572,
            "mean": 74.59296314824078,
            "std": 123.21206290146024,
            "variance": 15181.212444433397,
            "min": 0,
            "max": 307,
            "kurtosis": -0.6290387824541743,
            "skewness": 1.1256701172233239,
            "sum": 894668,
            "mad": 0.0,
            "range": 307,
            "5%": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 187.0,
            "95%": 307.0,
            "iqr": 187.0,
            "cv": 1.6517920417854604,
            "p_zeros": 0.7146906786726697,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Gender": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.7773886943471735,
            "std": 0.6830625785943093,
            "variance": 0.46657448627590703,
            "min": 1,
            "max": 3,
            "kurtosis": -0.8682689723647328,
            "skewness": 0.3138180442669538,
            "sum": 21318,
            "mad": 1.0,
            "range": 2,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 2.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 1.0,
            "cv": 0.38430680962848984,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Color1": {
            "n_distinct": 7,
            "p_distinct": 0.0005836251459062865,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 2.2389528097382025,
            "std": 1.7508240664915704,
            "variance": 3.065384911806079,
            "min": 1,
            "max": 7,
            "kurtosis": 0.9701360640824226,
            "skewness": 1.4644120191370003,
            "sum": 26854,
            "mad": 1.0,
            "range": 6,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 2.0,
            "75%": 3.0,
            "95%": 6.0,
            "iqr": 2.0,
            "cv": 0.7819834606948646,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Color2": {
            "n_distinct": 7,
            "p_distinct": 0.0005836251459062865,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 3600,
            "mean": 3.2131065532766385,
            "std": 2.74560357697596,
            "variance": 7.538339001903187,
            "min": 0,
            "max": 7,
            "kurtosis": -1.5098445593078604,
            "skewness": 0.19728631311033176,
            "sum": 38538,
            "mad": 2.0,
            "range": 7,
            "5%": 0.0,
            "25%": 0.0,
            "50%": 2.0,
            "75%": 6.0,
            "95%": 7.0,
            "iqr": 6.0,
            "cv": 0.8545012533667979,
            "p_zeros": 0.3001500750375188,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Color3": {
            "n_distinct": 6,
            "p_distinct": 0.0005002501250625312,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 8515,
            "mean": 1.8660163415040854,
            "std": 2.9779154359383995,
            "variance": 8.867980343600188,
            "min": 0,
            "max": 7,
            "kurtosis": -0.8719163439987785,
            "skewness": 1.0242005667700744,
            "sum": 22381,
            "mad": 0.0,
            "range": 7,
            "5%": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 5.0,
            "95%": 7.0,
            "iqr": 5.0,
            "cv": 1.595867822646225,
            "p_zeros": 0.7099383024845756,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "MaturitySize": {
            "n_distinct": 4,
            "p_distinct": 0.00033350008337502084,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.861347340336835,
            "std": 0.5451682527705674,
            "variance": 0.2972084238289132,
            "min": 1,
            "max": 4,
            "kurtosis": 0.4605567339538248,
            "skewness": -0.0022545629547571404,
            "sum": 22325,
            "mad": 0.0,
            "range": 3,
            "5%": 1.0,
            "25%": 2.0,
            "50%": 2.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 0.0,
            "cv": 0.29288904921523784,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "FurLength": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.4688177422044355,
            "std": 0.5992600732036002,
            "variance": 0.35911263533598425,
            "min": 1,
            "max": 3,
            "kurtosis": -0.21261398750180094,
            "skewness": 0.8873347115703182,
            "sum": 17617,
            "mad": 0.0,
            "range": 2,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 1.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 1.0,
            "cv": 0.4079880409833673,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Vaccinated": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.7356178089044523,
            "std": 0.6693143222526211,
            "variance": 0.4479816619724855,
            "min": 1,
            "max": 3,
            "kurtosis": -0.8031690089363783,
            "skewness": 0.36489541999744696,
            "sum": 20817,
            "mad": 1.0,
            "range": 2,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 2.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 1.0,
            "cv": 0.3856346246384175,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Dewormed": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.5581123895280973,
            "std": 0.69757926564668,
            "variance": 0.4866168318601614,
            "min": 1,
            "max": 3,
            "kurtosis": -0.5278377808281949,
            "skewness": 0.8528579873444425,
            "sum": 18688,
            "mad": 0.0,
            "range": 2,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 1.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 1.0,
            "cv": 0.44770792552259636,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Sterilized": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.9144572286143071,
            "std": 0.5679268514311735,
            "variance": 0.32254090857652623,
            "min": 1,
            "max": 3,
            "kurtosis": 0.027778855084581444,
            "skewness": -0.011743205284064464,
            "sum": 22962,
            "mad": 0.0,
            "range": 2,
            "5%": 1.0,
            "25%": 2.0,
            "50%": 2.0,
            "75%": 2.0,
            "95%": 3.0,
            "iqr": 0.0,
            "cv": 0.29665162686462393,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Health": {
            "n_distinct": 3,
            "p_distinct": 0.0002501250625312656,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.0363515090878772,
            "std": 0.19925458145199001,
            "variance": 0.03970238822960773,
            "min": 1,
            "max": 3,
            "kurtosis": 36.3757011367912,
            "skewness": 5.813841143851706,
            "sum": 12430,
            "mad": 0.0,
            "range": 2,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 1.0,
            "75%": 1.0,
            "95%": 1.0,
            "iqr": 0.0,
            "cv": 0.19226544247266036,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Quantity": {
            "n_distinct": 19,
            "p_distinct": 0.001584125396031349,
            "is_unique": False,
            "n_unique": 1,
            "p_unique": 8.337502084375521e-05,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 1.5886276471569118,
            "std": 1.508592604952439,
            "variance": 2.2758516477171855,
            "min": 1,
            "max": 20,
            "kurtosis": 36.05480385473918,
            "skewness": 4.736983141694582,
            "sum": 19054,
            "mad": 0.0,
            "range": 19,
            "5%": 1.0,
            "25%": 1.0,
            "50%": 1.0,
            "75%": 1.0,
            "95%": 5.0,
            "iqr": 0.0,
            "cv": 0.9496200117455418,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Fee": {
            "n_distinct": 71,
            "p_distinct": 0.00591962647990662,
            "is_unique": False,
            "n_unique": 23,
            "p_unique": 0.00191762547940637,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 10108,
            "mean": 21.39803234950809,
            "std": 79.7781620492847,
            "variance": 6364.555139961931,
            "min": 0,
            "max": 3000,
            "kurtosis": 217.67204897705238,
            "skewness": 9.610892401231167,
            "sum": 256648,
            "mad": 0.0,
            "range": 3000,
            "5%": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "95%": 150.0,
            "iqr": 0.0,
            "cv": 3.7282943004392033,
            "p_zeros": 0.8427547106886777,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "State": {
            "n_distinct": 14,
            "p_distinct": 0.001167250291812573,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 0,
            "mean": 41345.91270635318,
            "std": 32.389300021821896,
            "variance": 1049.0667559035917,
            "min": 41324,
            "max": 41415,
            "kurtosis": -0.7659264365139249,
            "skewness": 1.0996311768801166,
            "sum": 495902877,
            "mad": 0.0,
            "range": 91,
            "5%": 41326.0,
            "25%": 41326.0,
            "50%": 41326.0,
            "75%": 41401.0,
            "95%": 41401.0,
            "iqr": 75.0,
            "cv": 0.0007833736856133057,
            "p_zeros": 0.0,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "RescuerID": {
            "n_distinct": 4789,
            "p_distinct": 0.3992829748207437,
            "is_unique": False,
            "n_unique": 3315,
            "p_unique": 0.2763881940970485,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "max_length": 32,
            "mean_length": 32.0,
            "median_length": 32,
            "min_length": 32,
            "n_characters_distinct": 16,
            "n_characters": 383808,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "VideoAmt": {
            "n_distinct": 9,
            "p_distinct": 0.0007503751875937969,
            "is_unique": False,
            "n_unique": 2,
            "p_unique": 0.00016675004168751042,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 11535,
            "mean": 0.05694513923628481,
            "std": 0.3443516269990526,
            "variance": 0.11857804301689463,
            "min": 0,
            "max": 8,
            "kurtosis": 114.37392383368731,
            "skewness": 9.129144018384311,
            "sum": 683,
            "mad": 0.0,
            "range": 8,
            "5%": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "95%": 0.0,
            "iqr": 0.0,
            "cv": 6.04707674118102,
            "p_zeros": 0.9617308654327164,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "Description": {
            "n_distinct": 11285,
            "p_distinct": 0.9416722296395194,
            "is_unique": False,
            "n_unique": 11041,
            "p_unique": 0.9213117489986649,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 10,
            "n": 11994,
            "p_missing": 0.0008337502084375521,
            "count": 11984,
            "memory_size": 96080,
            "max_length": 6664,
            "mean_length": 338.8663217623498,
            "median_length": 1377,
            "min_length": 1,
            "n_characters_distinct": 1448,
            "n_characters": 4060974,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "PetID": {
            "n_distinct": 11994,
            "p_distinct": 1.0,
            "is_unique": True,
            "n_unique": 11994,
            "p_unique": 1.0,
            "type": "Text",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "max_length": 9,
            "mean_length": 9.0,
            "median_length": 9,
            "min_length": 9,
            "n_characters_distinct": 16,
            "n_characters": 107946,
            "n_block_alias": 1,
            "n_scripts": 1,
            "n_category": 1,
            "cast_type": "None"
          },
          "PhotoAmt": {
            "n_distinct": 31,
            "p_distinct": 0.0025846256461564115,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 273,
            "mean": 3.8907787226946806,
            "std": 3.4946436139785306,
            "variance": 12.212533988720924,
            "min": 0.0,
            "max": 30.0,
            "kurtosis": 12.896628429989528,
            "skewness": 2.8883393469750502,
            "sum": 46666.0,
            "mad": 2.0,
            "range": 30.0,
            "5%": 1.0,
            "25%": 2.0,
            "50%": 3.0,
            "75%": 5.0,
            "95%": 10.0,
            "iqr": 3.0,
            "cv": 0.8981861635035893,
            "p_zeros": 0.02276138069034517,
            "p_infinite": 0.0,
            "monotonic_increase": False,
            "monotonic_decrease": False,
            "monotonic_increase_strict": False,
            "monotonic_decrease_strict": False,
            "monotonic": 0,
            "cast_type": "None"
          },
          "AdoptionSpeed": {
            "n_distinct": 5,
            "p_distinct": 0.00041687510421877606,
            "is_unique": False,
            "n_unique": 0,
            "p_unique": 0.0,
            "type": "Numeric",
            "hashable": True,
            "ordering": True,
            "n_missing": 0,
            "n": 11994,
            "p_missing": 0.0,
            "count": 11994,
            "memory_size": 96080,
            "n_negative": 0,
            "p_negative": 0.0,
            "n_infinite": 0,
            "n_zeros": 328,
            "mean": 2.516341504085376,
            "std": 1.1772495657598065,
            "variance": 1.385916540081653,
            "min": 0,
            "max": 4,
            "kurtosis": -1.1393137463852208,
            "skewness": -0.1549212733703575,
            "sum": 30181,
            "mad": 1.0,
            "range": 4,
            "5%": 1.0,
            "25%": 2.0,
            "50%": 2.0,
            "75%": 4.0,
            "95%": 4.0,
            "iqr": 2.0,
            "cv": 0.4678417312787224,
            "p_zeros": 0.02734700683675171,
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
        "description_summary": "This dataset contains information about pets listed on PetFinder.my, including tabular data, text descriptions, and image metadata. The goal is to predict the 'AdoptionSpeed' of pets, which indicates how quickly a pet is adopted, to help improve pet profiles and reduce animal suffering.",
        "note": "The target variable 'AdoptionSpeed' is an ordinal categorical variable with 5 possible ratings (0-4). The primary evaluation metric is quadratic weighted kappa, which measures agreement between actual and predicted ratings.",
        "task_type": "multi_class_classification",
        "target_columns": [
          "AdoptionSpeed"
        ],
        "evaluation_metric": "quadratic_weighted_kappa"
      }
    }

def preprocess_data(train_df, test_df, dataset_metadata_json, base_path, is_training=True):
    """
    Performs data cleaning and preprocessing.
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        dataset_metadata_json (dict): Metadata about the dataset.
        base_path (Path): Base path to the dataset.
        is_training (bool): True if preprocessing for training, False for prediction.
                            Affects whether preprocessors are fitted or just transformed.
    Returns:
        tuple: (X_train_processed, y_train_processed, X_test_processed, scaler, encoder, tfidf_vectorizers, image_feature_extractor)
               or (X_test_processed, None, None, None, None, None, None) if not training.
    """
    numerical_cols = []
    categorical_cols = []
    text_cols = []
    id_cols = []
    target_col = dataset_metadata_json['task_definition']['target_columns'][0]

    for col, meta in dataset_metadata_json['profiling_summary']['variables'].items():
        if col == target_col:
            continue
        if meta['type'] == 'Numeric':
            if meta['n_distinct'] < 20 and meta['n_distinct'] / meta['n'] < 0.01:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif meta['type'] == 'Text':
            if meta['is_unique'] or col.endswith('ID'):
                id_cols.append(col)
            elif meta['mean_length'] > 50 or col == 'Description':
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    # Handle 'Name' as a text column
    if 'Name' in categorical_cols:
        categorical_cols.remove('Name')
        text_cols.append('Name')

    # Ensure target column is not in feature lists
    if target_col in numerical_cols: numerical_cols.remove(target_col)
    if target_col in categorical_cols: categorical_cols.remove(target_col)
    if target_col in text_cols: text_cols.remove(target_col)

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Text columns: {text_cols}")
    print(f"ID columns: {id_cols}")
    print(f"Target column: {target_col}")

    # Store original PetIDs for submission
    train_pet_ids_original = train_df['PetID'] if 'PetID' in train_df.columns else None
    test_pet_ids_original = test_df['PetID'] if 'PetID' in test_df.columns else None

    # Impute missing values
    numerical_imputer = SimpleImputer(strategy='median')
    if is_training:
        train_df[numerical_cols] = numerical_imputer.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = numerical_imputer.transform(test_df[numerical_cols])

    for col in categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        if is_training:
            train_df[col] = categorical_imputer.fit_transform(train_df[[col]])
        test_df[col] = categorical_imputer.transform(test_df[[col]])

    for col in text_cols:
        train_df[col].fillna('', inplace=True)
        test_df[col].fillna('', inplace=True)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    if is_training:
        encoded_train_cols = encoder.fit_transform(train_df[categorical_cols])
    encoded_test_cols = encoder.transform(test_df[categorical_cols])

    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    if is_training:
        encoded_train_df = pd.DataFrame(encoded_train_cols, columns=encoded_feature_names, index=train_df.index)
        train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_train_df], axis=1)
    encoded_test_df = pd.DataFrame(encoded_test_cols, columns=encoded_feature_names, index=test_df.index)
    test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

    # TF-IDF for text columns
    tfidf_vectorizers = {}
    for col in text_cols:
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        if is_training:
            train_text_features = tfidf.fit_transform(train_df[col])
        test_text_features = tfidf.transform(test_df[col])

        if is_training:
            train_text_df = pd.DataFrame(train_text_features.toarray(), columns=[f'{col}_tfidf_{i}' for i in range(train_text_features.shape[1])], index=train_df.index)
            train_df = pd.concat([train_df.drop(columns=[col]), train_text_df], axis=1)
        test_text_df = pd.DataFrame(test_text_features.toarray(), columns=[f'{col}_tfidf_{i}' for i in range(test_text_features.shape[1])], index=test_df.index)
        test_df = pd.concat([test_df.drop(columns=[col]), test_text_df], axis=1)
        tfidf_vectorizers[col] = tfidf

    # Scale numerical features
    scaler = StandardScaler()
    if is_training:
        train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    # Image Preprocessing (Feature Extraction)
    TRAIN_IMAGE_DIR = base_path / 'train_images'
    TEST_IMAGE_DIR = base_path / 'test_images'

    image_feature_extractor = None
    try:
        image_feature_extractor = hub.KerasLayer(
            "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
            trainable=False
        )
    except Exception as e:
        print(f"Could not load TensorFlow Hub model. Skipping image feature extraction: {e}")

    if TRAIN_IMAGE_DIR.exists() and TEST_IMAGE_DIR.exists() and image_feature_extractor is not None:
        if is_training:
            print("Extracting image features for training data...")
            train_image_features_df = extract_image_features(train_df[['PetID', 'PhotoAmt']], TRAIN_IMAGE_DIR, image_feature_extractor)
            train_df = train_df.set_index('PetID').join(train_image_features_df, how='left').reset_index()

        print("Extracting image features for test data...")
        test_image_features_df = extract_image_features(test_df[['PetID', 'PhotoAmt']], TEST_IMAGE_DIR, image_feature_extractor)
        test_df = test_df.set_index('PetID').join(test_image_features_df, how='left').reset_index()
    else:
        print("Image directories not found or TensorFlow Hub model not loaded. Skipping image feature extraction.")
        # Add dummy image feature columns if they are expected by the model
        dummy_img_cols = [f'img_feat_{i}' for i in range(1280)]
        if is_training:
            for col in dummy_img_cols:
                train_df[col] = 0.0
        for col in dummy_img_cols:
            test_df[col] = 0.0

    # Drop PetID and RescuerID
    if 'PetID' in train_df.columns:
        train_df = train_df.drop(columns=['PetID'])
    if 'PetID' in test_df.columns:
        test_df = test_df.drop(columns=['PetID'])

    if 'RescuerID' in train_df.columns:
        train_df = train_df.drop(columns=['RescuerID'])
    if 'RescuerID' in test_df.columns:
        test_df = test_df.drop(columns=['RescuerID'])

    # Align columns between train and test
    if is_training:
        X_train_full = train_df.drop(columns=[target_col])
        y_train_full = train_df[target_col]
    else:
        X_train_full = None # Not needed for prediction path

    # Get columns from the training set (or a reference set if only predicting)
    # If is_training is False, we need a reference for columns.
    # This is a common issue in deployment; ideally, the column names from training
    # would be saved and loaded. For this refactoring, we'll assume `train_df`
    # (even if it's just a dummy for column alignment) is available.
    # A more robust solution would save `X_train_full.columns` during training.
    if is_training:
        reference_cols = X_train_full.columns
    else:
        # For prediction, we need to load the training data just to get column names
        # and fit preprocessors. This is not ideal for production but matches the original logic.
        temp_train_df, _, _, _, _, _ = load_data(base_path)
        # Re-run a minimal preprocessing on temp_train_df to get column names
        # This is a hacky way to get the column names without re-fitting everything.
        # A better way is to save the column names during training.
        # For now, we'll just use the current train_df's columns.
        # This assumes that the `train_df` passed to this function (even if it's just for column alignment)
        # has the correct structure.
        # Let's assume `train_df` passed to this function *is* the full training data for fitting.
        # If `is_training` is False, `train_df` is used only for fitting preprocessors.
        # So, we need to ensure `train_df` is the full training data when `is_training` is False.
        # This implies `main_pipeline` will pass the full training data to `preprocess_data`
        # even when only predicting.
        # To avoid this, we should save the fitted preprocessors and column names.
        # For this refactoring, we'll stick to the original logic of re-fitting preprocessors
        # on the full training data when predicting.
        # This means `train_df` should be the full training data when `is_training` is False.
        # The `main_pipeline` function will handle this.
        reference_cols = train_df.drop(columns=[target_col]).columns


    # Add missing columns to test_df, fill with 0
    missing_in_test = list(set(reference_cols) - set(test_df.columns))
    for col in missing_in_test:
        test_df[col] = 0

    # Remove extra columns from test_df that are not in reference_cols
    extra_in_test = list(set(test_df.columns) - set(reference_cols))
    test_df = test_df.drop(columns=extra_in_test)

    # Ensure column order is the same
    test_df_processed = test_df[list(reference_cols)]

    print("\nPreprocessing complete.")
    if is_training:
        print(f"Train data shape: {X_train_full.shape}")
    print(f"Test data shape: {test_df_processed.shape}")

    if is_training:
        return X_train_full, y_train_full, test_df_processed, numerical_imputer, encoder, tfidf_vectorizers, scaler, image_feature_extractor, train_pet_ids_original, test_pet_ids_original
    else:
        return test_df_processed, test_pet_ids_original

def train_model(X_train_full, y_train_full, root_dir):
    """
    Trains the LightGBM model and saves it along with evaluation metrics.
    """
    OUTPUTS_DIR = root_dir / 'outputs'
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH = OUTPUTS_DIR / "metrics.json"
    MODEL_PATH = root_dir / "models/pet_finder_model.pkl"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Perform an 80/20 stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"\nTraining data split into X_train: {X_train.shape}, X_val: {X_val.shape}")

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMClassifier(**lgb_params)

    print("\nStarting model training...")
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    print("Model training complete.")

    # Evaluation
    print("\nEvaluating model on validation set...")
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_val, y_pred_val)
    metrics['f1_macro'] = f1_score(y_val, y_pred_val, average='macro')
    metrics['log_loss'] = log_loss(y_val, y_proba_val)

    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_val, y_proba_val, multi_class='ovr')
    except ValueError as e:
        metrics['roc_auc_ovr'] = f"Not applicable or error: {e}"
        print(f"Warning: Could not calculate ROC_AUC_OVR: {e}")

    metrics['quadratic_weighted_kappa'] = cohen_kappa_score(y_val, y_pred_val, weights='quadratic')

    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Model Persistence
    print(f"\nSaving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")

    return model

def generate_predictions(trained_model, X_test_processed, test_pet_ids_original, root_dir):
    """
    Generates predictions using the trained model and saves the submission file.
    """
    OUTPUTS_DIR = root_dir / 'outputs'
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

    print("\nGenerating predictions...")
    predictions = trained_model.predict(X_test_processed)

    submission_df = pd.DataFrame({'PetID': test_pet_ids_original, 'AdoptionSpeed': predictions})

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file saved to {SUBMISSION_PATH}")

def main_pipeline():
    """
    Main function to orchestrate the entire ML pipeline:
    1. Load data and metadata.
    2. Preprocess data (train and test).
    3. Train the model.
    4. Evaluate the model.
    5. Generate predictions and submission file.
    """
    ROOT_DIR, BASE_PATH = get_base_paths()
    dataset_metadata_json = get_dataset_metadata()

    # Load datasets
    train_df, test_df, _, _, _, sample_submission_df = load_data(BASE_PATH)

    # Preprocess data for training
    # Note: We pass both train_df and test_df to preprocess_data,
    # and it will fit preprocessors on train_df and transform both.
    # This is consistent with the original concatenated scripts.
    X_train_full, y_train_full, X_test_processed_for_training, numerical_imputer, encoder, tfidf_vectorizers, scaler, image_feature_extractor, train_pet_ids, test_pet_ids = \
        preprocess_data(train_df.copy(), test_df.copy(), dataset_metadata_json, BASE_PATH, is_training=True)

    print("\nPreprocessing for training complete.")
    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"y_train_full shape: {y_train_full.shape}")
    print(f"X_test_processed_for_training shape: {X_test_processed_for_training.shape}")

    # Train the model
    trained_model = train_model(X_train_full, y_train_full, ROOT_DIR)

    # Generate predictions on the preprocessed test data
    generate_predictions(trained_model, X_test_processed_for_training, test_pet_ids, ROOT_DIR)

if __name__ == '__main__':
    main_pipeline()