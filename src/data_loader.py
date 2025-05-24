import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from src import config

def load_and_preprocess_data(for_transformer=True):
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {config.DATA_PATH}")
        print("Please ensure 'fetal_health.csv' (from UCI or Kaggle) is in the 'data/' directory.")
        return None, None, None, None, None, None, None, None, None

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=3)
    class_names = ['Normal', 'Suspect', 'Pathological']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_categorical,
        test_size=config.TEST_SPLIT,
        stratify=y_categorical,
        random_state=config.RANDOM_SEED
    )

    val_split_adjusted = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_split_adjusted,
        stratify=y_train_val,
        random_state=config.RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if for_transformer:
        X_train_reshaped = X_train_scaled[:, :, np.newaxis]
        X_val_reshaped = X_val_scaled[:, :, np.newaxis]
        X_test_reshaped = X_test_scaled[:, :, np.newaxis]
    else:
        X_train_reshaped = X_train_scaled
        X_val_reshaped = X_val_scaled
        X_test_reshaped = X_test_scaled

    print(f"Training data shape: {X_train_reshaped.shape}, Labels: {y_train.shape}")
    print(f"Validation data shape: {X_val_reshaped.shape}, Labels: {y_val.shape}")
    print(f"Test data shape: {X_test_reshaped.shape}, Labels: {y_test.shape}")

    num_features = X_train_reshaped.shape[1]
    num_classes = y_train.shape[1]

    return (X_train_reshaped, y_train, X_val_reshaped, y_val, X_test_reshaped, y_test,
            scaler, num_features, num_classes, class_names)

if __name__ == '__main__':
    load_and_preprocess_data()
