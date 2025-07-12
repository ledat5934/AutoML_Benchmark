import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
import joblib
from tqdm import tqdm
import warnings

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Determine the project root
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:  # __file__ is not defined inside Kaggle/Jupyter
    ROOT_DIR = Path.cwd()

# Define BASE_PATH with fallback
BASE_PATH_CANDIDATE_1 = (ROOT_DIR / 'input/Datasets/datasets/multi_label_classification').resolve()
BASE_PATH_CANDIDATE_2 = Path('input/Datasets/datasets/multi_label_classification').resolve()

if BASE_PATH_CANDIDATE_1.exists():
    BASE_PATH = BASE_PATH_CANDIDATE_1
else:
    BASE_PATH = BASE_PATH_CANDIDATE_2

print(f"Resolved BASE_PATH: {BASE_PATH}")

# File path constants
TRAIN_CSV_PATH = BASE_PATH / 'train.csv'
TEST_CSV_PATH = BASE_PATH / 'test.csv'
IMAGE_FOLDER_PATH = BASE_PATH / 'data'
METADATA_JSON_PATH = BASE_PATH.parent.parent / 'multi_label_classification.json'

# Output paths
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
MODEL_PATH = Path("./models/multi_label_classification_model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
MLB_PATH = Path("./models/mlb.pkl")
TFIDF_PATH = Path("./models/tfidf_vectorizer.pkl")

# Image preprocessing constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
EARLY_STOPPING_ROUNDS = 5
LEARNING_RATE = 1e-4

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class MultiLabelImageDataset(Dataset):
    def __init__(self, dataframe, image_folder_path, transform=None, is_test=False):
        self.dataframe = dataframe
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.is_test = is_test
        self.label_columns = [col for col in dataframe.columns if col.isdigit()]
        self.tfidf_columns = [col for col in dataframe.columns if col.startswith('caption_tfidf_')]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['ImageID']
        img_path = os.path.join(self.image_folder_path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tfidf_features = torch.tensor(self.dataframe.iloc[idx][self.tfidf_columns].values.astype(np.float32))

        if self.is_test:
            return image, tfidf_features, img_name
        else:
            labels = self.dataframe.iloc[idx][self.label_columns].values.astype(np.float32)
            return image, torch.tensor(labels), tfidf_features, img_name

class MultiModalModel(nn.Module):
    def __init__(self, num_classes, tfidf_features_dim):
        super(MultiModalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.tfidf_fc = nn.Linear(tfidf_features_dim, 512)

        self.combined_fc1 = nn.Linear(num_ftrs + 512, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.combined_fc2 = nn.Linear(1024, num_classes)

    def forward(self, image_input, tfidf_input):
        image_features = self.resnet(image_input)
        tfidf_features = self.relu(self.tfidf_fc(tfidf_input))
        combined_features = torch.cat((image_features, tfidf_features), dim=1)
        output = self.combined_fc1(combined_features)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.combined_fc2(output)
        return output

def load_metadata(metadata_path):
    """Loads and returns the dataset metadata JSON."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def preprocess_data(df, metadata, mlb=None, tfidf_vectorizer=None, scaler=None, is_train=True):
    """
    Performs preprocessing steps on the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe (train or test).
        metadata (dict): The dataset metadata.
        mlb (MultiLabelBinarizer, optional): Fitted MultiLabelBinarizer for labels.
                                             Required for test set.
        tfidf_vectorizer (TfidfVectorizer, optional): Fitted TfidfVectorizer for captions.
                                                      Required for test set.
        scaler (StandardScaler, optional): Fitted StandardScaler for numerical features.
                                           Required for test set.
        is_train (bool): True if processing the training set, False for test set.
    Returns:
        tuple: Processed dataframe, fitted mlb, fitted tfidf_vectorizer, fitted scaler.
    """
    variables_info = metadata['profiling_summary']['variables']
    image_id_col = None
    label_col = None
    caption_col = None

    for col_name, col_info in variables_info.items():
        if col_name == 'ImageID':
            image_id_col = col_name
        elif col_name == 'Labels':
            label_col = col_name
        elif col_name == 'Caption':
            caption_col = col_name

    if label_col and label_col in df.columns:
        df[label_col] = df[label_col].apply(lambda x: x.split())

        if is_train:
            mlb = MultiLabelBinarizer()
            labels_encoded = mlb.fit_transform(df[label_col])
            labels_df = pd.DataFrame(labels_encoded, columns=mlb.classes_, index=df.index)
            df = pd.concat([df.drop(columns=[label_col]), labels_df], axis=1)
        else:
            if mlb is None:
                raise ValueError("MultiLabelBinarizer must be fitted on training data and provided for test data.")
            if label_col in df.columns:
                labels_encoded = mlb.transform(df[label_col])
                labels_df = pd.DataFrame(labels_encoded, columns=mlb.classes_, index=df.index)
                df = pd.concat([df.drop(columns=[label_col]), labels_df], axis=1)
            else:
                print(f"Label column '{label_col}' not found in test dataframe. Skipping label binarization.")
    else:
        print(f"Warning: Label column '{label_col}' not found or not specified in metadata. Skipping label processing.")

    if caption_col and caption_col in df.columns:
        if is_train:
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            caption_features = tfidf_vectorizer.fit_transform(df[caption_col])
        else:
            if tfidf_vectorizer is None:
                raise ValueError("TfidfVectorizer must be fitted on training data and provided for test data.")
            caption_features = tfidf_vectorizer.transform(df[caption_col])

        caption_df = pd.DataFrame(caption_features.toarray(),
                                  columns=[f'caption_tfidf_{i}' for i in range(caption_features.shape[1])],
                                  index=df.index)
        df = pd.concat([df.drop(columns=[caption_col]), caption_df], axis=1)
    else:
        print(f"Warning: Caption column '{caption_col}' not found or not specified in metadata. Skipping TF-IDF processing.")

    return df, mlb, tfidf_vectorizer, scaler

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, early_stopping_rounds):
    best_val_f1 = -1
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []

        for images, labels, tfidf_features, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            tfidf_features = tfidf_features.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, tfidf_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.append(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        all_train_preds = np.vstack(train_preds)
        all_train_targets = np.vstack(train_targets)
        train_binary_preds = (all_train_preds > 0.5).astype(int)
        epoch_train_f1 = f1_score(all_train_targets, train_binary_preds, average='samples')

        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, labels, tfidf_features, _ in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                tfidf_features = tfidf_features.to(DEVICE)

                outputs = model(images, tfidf_features)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        all_val_preds = np.vstack(val_preds)
        all_val_targets = np.vstack(val_targets)
        val_binary_preds = (all_val_preds > 0.5).astype(int)
        epoch_val_f1 = f1_score(all_val_targets, val_binary_preds, average='samples')

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train F1: {epoch_train_f1:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val F1: {epoch_val_f1:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_f1'].append(epoch_val_f1)

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH} with improved F1: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_rounds:
                print(f"Early stopping triggered after {early_stopping_rounds} epochs without improvement.")
                break
    return model, history

def evaluate_model(model, dataloader, mlb):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels, tfidf_features, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            tfidf_features = tfidf_features.to(DEVICE)

            outputs = model(images, tfidf_features)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probabilities)
            all_targets.append(labels.cpu().numpy())

    predictions_array = np.vstack(all_preds)
    targets_array = np.vstack(all_targets)

    binary_predictions = (predictions_array > 0.5).astype(int)

    f1 = f1_score(targets_array, binary_predictions, average='samples')
    accuracy = accuracy_score(targets_array, binary_predictions)
    loss = log_loss(targets_array.ravel(), predictions_array.ravel())
    if targets_array.shape[1] > 1:
        roc_auc = roc_auc_score(targets_array, predictions_array, average='macro')
    else:
        roc_auc = roc_auc_score(targets_array, predictions_array)

    metrics = {
        "f1_score": f1,
        "accuracy": accuracy,
        "log_loss": loss,
        "roc_auc_score": roc_auc
    }
    return metrics

def generate_predictions(model, test_dataloader, mlb):
    model.eval()
    all_predictions = []
    image_ids = []

    with torch.no_grad():
        for images, tfidf_features, img_names in tqdm(test_dataloader, desc="Generating predictions"):
            images = images.to(DEVICE)
            tfidf_features = tfidf_features.to(DEVICE)

            outputs = model(images, tfidf_features)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(probabilities)
            image_ids.extend(img_names)

    predictions_array = np.vstack(all_predictions)
    binary_predictions = (predictions_array > 0.5).astype(int)

    predicted_labels = []
    for i in range(binary_predictions.shape[0]):
        row_labels = mlb.inverse_transform(binary_predictions[i:i+1])
        predicted_labels.append(" ".join(row_labels[0]))

    submission_df = pd.DataFrame({'ImageID': image_ids, 'Labels': predicted_labels})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file generated successfully at: {SUBMISSION_PATH}")

def main():
    metadata = load_metadata(METADATA_JSON_PATH)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    print("Original Train DataFrame head:")
    print(train_df.head())
    print("\nOriginal Test DataFrame head:")
    print(test_df.head())

    processed_train_df, mlb, tfidf_vectorizer, scaler = preprocess_data(train_df.copy(), metadata, is_train=True)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)

    label_cols = [col for col in processed_train_df.columns if col.isdigit()]
    label_combinations = processed_train_df[label_cols].apply(lambda row: tuple(row), axis=1)
    combination_counts = label_combinations.value_counts()
    rare_combinations = combination_counts[combination_counts < 2].index

    stratify_target = label_combinations.apply(lambda x: str(x))
    stratify_target[label_combinations.isin(rare_combinations).values] = 'RARE_COMBINATION'

    train_idx, val_idx = train_test_split(
        processed_train_df.index,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

    train_df_split = processed_train_df.loc[train_idx].reset_index(drop=True)
    val_df_split = processed_train_df.loc[val_idx].reset_index(drop=True)

    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_image_dataset = MultiLabelImageDataset(train_df_split, IMAGE_FOLDER_PATH, transform=image_transform, is_test=False)
    val_image_dataset = MultiLabelImageDataset(val_df_split, IMAGE_FOLDER_PATH, transform=image_transform, is_test=False)

    train_dataloader = DataLoader(train_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_dataloader = DataLoader(val_image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    print(f"\nTrain image dataset size: {len(train_image_dataset)}")
    print(f"Test image dataset size: {len(val_image_dataset)}") # This is actually validation set
    print(f"Example image batch from train_image_dataloader (shape):")
    for i, (images, labels, tfidf_features, img_ids) in enumerate(train_dataloader):
        print(f"Batch {i+1}: Images shape: {images.shape}, Labels shape: {labels.shape}, TF-IDF shape: {tfidf_features.shape}, Image IDs: {img_ids[:5]}")
        break

    num_classes = len(mlb.classes_)
    tfidf_features_dim = len(tfidf_vectorizer.vocabulary_)

    model = MultiModalModel(num_classes=num_classes, tfidf_features_dim=tfidf_features_dim).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model, history = train_model(model, train_dataloader, val_dataloader, optimizer, criterion, NUM_EPOCHS, EARLY_STOPPING_ROUNDS)

    trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    val_metrics = evaluate_model(trained_model, val_dataloader, mlb)

    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    with open(METRICS_PATH, 'w') as f:
        json.dump(val_metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")

    # --- Stage 3: Prediction ---
    # Load preprocessors (already loaded above, but re-loading for clarity if this were a separate run)
    mlb = joblib.load(MLB_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)

    # Preprocess test data
    processed_test_df, _, _, _ = preprocess_data(test_df.copy(), metadata, mlb=mlb, tfidf_vectorizer=tfidf_vectorizer, scaler=None, is_train=False)

    test_image_dataset = MultiLabelImageDataset(processed_test_df, IMAGE_FOLDER_PATH, transform=image_transform, is_test=True)
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    # Load the trained model (already loaded and evaluated, but re-loading for clarity if this were a separate run)
    num_classes = len(mlb.classes_)
    tfidf_features_dim = len(tfidf_vectorizer.vocabulary_)
    model_for_prediction = MultiModalModel(num_classes=num_classes, tfidf_features_dim=tfidf_features_dim)
    model_for_prediction.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_for_prediction.to(DEVICE)

    generate_predictions(model_for_prediction, test_image_dataloader, mlb)

if __name__ == "__main__":
    main()