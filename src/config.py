# FILE: src/config.py

import os
import tensorflow as tf

DATA_PATH = os.path.join("data", "fetal_health.csv")
MODEL_SAVE_PATH = os.path.join("models", "best_fetal_health_hybrid_model.keras")
OPTUNA_DB_PATH = f"sqlite:///{os.path.join('models', 'optuna_study.db')}"
RESULTS_DIR = "results"

TARGET_ACCURACY = 0.97
RANDOM_SEED = 42


BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

EPOCHS = 300
INITIAL_LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05

EARLY_STOPPING_PATIENCE = 50
REDUCE_LR_PATIENCE = 15
REDUCE_LR_FACTOR = 0.2

CONV_FILTERS_BASE = 64
CONV_KERNEL_SIZE = 5
CONV_LAYERS = 2
CONV_DROPOUT = 0.2

TRANSFORMER_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
DFF_MULTIPLIER = 4
TRANSFORMER_DROPOUT = 0.2

DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 128
HEAD_DROPOUT = 0.5

# --- Setup Directories and Seeds ---
os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "optuna_visualizations"), exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)
