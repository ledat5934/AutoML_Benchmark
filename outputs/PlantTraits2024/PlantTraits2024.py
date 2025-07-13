import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import warnings
from tqdm import tqdm

# Suppress specific warnings from LightGBM
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Determine the project root
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
except NameError:  # __file__ is not defined inside Kaggle/Jupyter
    ROOT_DIR = Path.cwd()

# Define base path with fallback
BASE_PATH = Path('/kaggle/input/planttraits2024')

print(f"Resolved BASE_PATH: {BASE_PATH}")

# File path constants
TRAIN_CSV_PATH = BASE_PATH / 'train.csv'
TEST_CSV_PATH = BASE_PATH / 'test.csv'
TARGET_NAME_META_PATH = BASE_PATH / 'target_name_meta.tsv'
SAMPLE_SUBMISSION_PATH = BASE_PATH / 'sample_submission.csv'
TRAIN_IMAGES_PATH = BASE_PATH / 'train_images'
TEST_IMAGES_PATH = BASE_PATH / 'test_images'
DATASET_METADATA_PATH = BASE_PATH / 'PlantTraits2024.json'

# Model and metrics paths
MODEL_PATH = Path("./models/PlantTraits2024_model.pth")
METRICS_PATH = Path("./outputs/metrics.json")

# Ensure output directories exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Prediction and submission paths
TEST_PROCESSED_PATH = Path("./processed/test_processed.csv")
SUBMISSION_PATH = Path("./outputs/submission.csv")
TEST_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# Image preprocessing constants
IMAGE_SIZE = (224, 224)

class PlantDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_train=True, target_columns=None, tabular_features_df=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.target_columns = target_columns if target_columns is not None else []
        self.tabular_features_df = tabular_features_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = int(self.dataframe.iloc[idx]['id'])
        img_name = f"{img_id}.jpeg"
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tabular_data_row = self.tabular_features_df.loc[img_id]
        tabular_data_tensor = torch.tensor(tabular_data_row.values.astype(np.float32))

        if self.is_train:
            targets = self.dataframe.iloc[idx][self.target_columns].values.astype(np.float32)
            return image, tabular_data_tensor, torch.tensor(targets)
        else:
            return image, tabular_data_tensor

class CombinedTestDataset(Dataset):
    def __init__(self, X_test_processed_df, img_dir, transform=None):
        self.X_test_processed_df = X_test_processed_df
        self.img_dir = img_dir
        self.transform = transform
        self.ids = X_test_processed_df.index.values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        current_id = self.ids[idx]

        img_name = f"{int(current_id)}.jpeg"
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tabular_data = self.X_test_processed_df.loc[current_id].values.astype(np.float32)
        tabular_data_tensor = torch.tensor(tabular_data)

        return image, tabular_data_tensor

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def preprocess_data():
    metadata = load_metadata(DATASET_METADATA_PATH)
    target_columns = metadata['task_definition']['target_columns']

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    X_train = train_df.drop(columns=target_columns)
    y_train = train_df[target_columns]
    X_test = test_df.copy()

    numerical_cols = []
    categorical_cols = []

    train_feature_cols_no_id = [col for col in X_train.columns if col != 'id']
    test_feature_cols_no_id = [col for col in X_test.columns if col != 'id']

    common_feature_cols = list(set(train_feature_cols_no_id) & set(test_feature_cols_no_id))

    for col in common_feature_cols:
        if col in metadata['profiling_summary']['variables']:
            var_info = metadata['profiling_summary']['variables'][col]
            if var_info['type'] == 'Numeric':
                numerical_cols.append(col)

    numerical_cols.sort()

    numerical_imputer = SimpleImputer(strategy='median')
    X_train_numerical_imputed = numerical_imputer.fit_transform(X_train[numerical_cols])
    X_test_numerical_imputed = numerical_imputer.transform(X_test[numerical_cols])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numerical_imputed)
    X_test_scaled = scaler.transform(X_test_numerical_imputed)

    X_train_processed = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train['id'])
    X_test_processed = pd.DataFrame(X_test_scaled, columns=numerical_cols, index=X_test['id'])

    y_train = y_train.set_index(train_df['id'])
    y_train_processed = y_train.loc[X_train_processed.index]

    return X_train_processed, y_train_processed, X_test_processed, numerical_cols, categorical_cols, scaler, numerical_imputer, target_columns

class MultiModalModel(nn.Module):
    def __init__(self, num_tabular_features, num_targets):
        super(MultiModalModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.tabular_fc = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )

    def forward(self, image_input, tabular_input):
        image_features = self.resnet(image_input)
        tabular_features = self.tabular_fc(tabular_input)

        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.fusion_fc(combined_features)
        return output

def main_logic(run_training=False):
    X_train_processed, y_train_processed, X_test_processed, numerical_cols, categorical_cols, scaler, numerical_imputer, target_columns = preprocess_data()

    print("\n--- Preprocessing Summary ---")
    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train_processed: {y_train_processed.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")
    print(f"Numerical columns processed: {numerical_cols[:5]}...")
    print(f"Categorical columns processed: {categorical_cols}")
    print(f"Missing values in X_train_processed after imputation: {X_train_processed.isnull().sum().sum()}")
    print(f"Missing values in X_test_processed after imputation: {X_test_processed.isnull().sum().sum()}")

    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_tabular_features = X_train_processed.shape[1]
    num_targets = y_train_processed.shape[1]

    model = MultiModalModel(num_tabular_features, num_targets).to(device)

    if run_training:
        train_combined_df = pd.concat([X_train_processed, y_train_processed], axis=1)
        train_ids = train_combined_df.index
        train_idx, val_idx = train_test_split(train_ids, test_size=0.2, random_state=42)

        original_train_df = pd.read_csv(TRAIN_CSV_PATH)

        train_image_df_split = original_train_df[original_train_df['id'].isin(train_idx)].reset_index(drop=True)
        val_image_df_split = original_train_df[original_train_df['id'].isin(val_idx)].reset_index(drop=True)

        X_train_split_tabular = X_train_processed.loc[train_idx]
        y_train_split = y_train_processed.loc[train_idx]
        X_val_split_tabular = X_train_processed.loc[val_idx]
        y_val_split = y_train_processed.loc[val_idx]

        print(f"Shape of X_train_split_tabular: {X_train_split_tabular.shape}")
        print(f"Shape of y_train_split: {y_train_split.shape}")
        print(f"Shape of X_val_split_tabular: {X_val_split_tabular.shape}")
        print(f"Shape of y_val_split: {y_val_split.shape}")

        train_dataset = PlantDataset(train_image_df_split, TRAIN_IMAGES_PATH, transform=image_transform, is_train=True, target_columns=target_columns, tabular_features_df=X_train_split_tabular)
        val_dataset = PlantDataset(val_image_df_split, TRAIN_IMAGES_PATH, transform=image_transform, is_train=True, target_columns=target_columns, tabular_features_df=X_val_split_tabular)

        BATCH_SIZE = 32
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print("\n--- Training Multi-Modal Model ---")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, tabular_data, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
                images, tabular_data, targets = images.to(device), tabular_data.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images, tabular_data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_dataloader.dataset)

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images, tabular_data, targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                    images, tabular_data, targets = images.to(device), tabular_data.to(device), targets.to(device)
                    outputs = model(images, tabular_data)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * images.size(0)

            val_epoch_loss = val_running_loss / len(val_dataloader.dataset)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    break

        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        val_predictions = []
        val_true = []
        with torch.no_grad():
            for images, tabular_data, targets in val_dataloader:
                images, tabular_data = images.to(device), tabular_data.to(device)
                outputs = model(images, tabular_data)
                val_predictions.append(outputs.cpu().numpy())
                val_true.append(targets.cpu().numpy())

        val_predictions = np.vstack(val_predictions)
        val_true = np.vstack(val_true)

        metrics = {}
        overall_r2 = []
        per_target_rmse = []             

        for i, target_col in enumerate(target_columns):
            y_true_target = val_true[:, i]
            y_pred_target = val_predictions[:, i]

            rmse = np.sqrt(mean_squared_error(y_true_target, y_pred_target))
            mae = mean_absolute_error(y_true_target, y_pred_target)
            r2  = r2_score(y_true_target, y_pred_target)

            metrics[target_col] = {
                "RMSE": float(rmse),
                "MAE":  float(mae),
                "R2":   float(r2)
            }
            per_target_rmse.append(rmse)  
            overall_r2.append(r2)

            print(f"\nMetrics for {target_col}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R2:   {r2:.4f}")

        overall_mse  = np.mean((val_predictions - val_true) ** 2)
        overall_rmse = np.sqrt(overall_mse)
        metrics["overall_RMSE"]            = float(overall_rmse)
        metrics["mean_per_target_RMSE"]    = float(np.mean(per_target_rmse))
        mean_r2_overall = np.mean(overall_r2)
        metrics["overall_mean_R2"]         = float(mean_r2_overall)

        print(f"\nOverall RMSE (flat): {overall_rmse:.4f}")
        print(f"Overall Mean R2:     {mean_r2_overall:.4f}")


        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {METRICS_PATH}")
        print(f"Trained model state saved to {MODEL_PATH}")

    # Prediction phase (always runs after potential training or if run_training is False)
    X_test_processed.to_csv(TEST_PROCESSED_PATH)
    print(f"Processed test features saved to {TEST_PROCESSED_PATH}")

    test_dataset = CombinedTestDataset(X_test_processed, TEST_IMAGES_PATH, transform=image_transform)
    BATCH_SIZE = 32
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Using device for prediction: {device}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_predictions = []
    with torch.no_grad():
        for images, tabular_data in tqdm(test_dataloader, desc="Predicting on Test Set"):
            images, tabular_data = images.to(device), tabular_data.to(device)
            outputs = model(images, tabular_data)
            test_predictions.append(outputs.cpu().numpy())

    test_predictions = np.vstack(test_predictions)

    submission_df = pd.DataFrame(test_predictions, columns=target_columns)
    submission_df.insert(0, 'id', X_test_processed.index.values)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file generated and saved to {SUBMISSION_PATH}")

    return submission_df

if __name__ == '__main__':
    # Set run_training to True to train the model, False to only run prediction using a pre-trained model
    submission_df = main_logic(run_training=True) # Set to True to run training and then prediction, False for prediction only