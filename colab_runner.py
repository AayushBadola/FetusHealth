# FILE: colab_runner.py

import tensorflow as tf
import os
import json
import datetime
import numpy as np

from src.data_loader import load_and_preprocess_data
from src.trainer import train_model
from src.evaluate import evaluate_model
from src.hyperparameter_tuner import run_hyperparameter_tuning
from src import config

def run_on_tpu(run_hpo=True, n_hpo_trials=70):
    print("--- Initializing TPU Strategy ---")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Successfully connected to TPU.")
        print(f"Number of replicas (cores): {strategy.num_replicas_in_sync}")
    except Exception as e:
        print(f"Could not connect to TPU: {e}")
        print("Falling back to default strategy (CPU/GPU).")
        strategy = tf.distribute.get_strategy()

    print(f"--- Starting Fetal Health Pipeline on {strategy.num_replicas_in_sync} Replicas ---")

    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    best_hps = None
    if run_hpo:
        print(f"--- Starting HPO phase for {n_hpo_trials} trials ---")
        best_hps = run_hyperparameter_tuning(n_trials=n_hpo_trials, strategy=strategy)
        if best_hps:
            print("HPO complete. Best HPs found:", best_hps)
            json_path = os.path.join(config.RESULTS_DIR, "best_optuna_params.json")
            with open(json_path, "w") as f:
                json.dump(best_hps, f, indent=4)
            print(f"Best hyperparameters saved to {json_path}")
    else:
        print("--- Skipping HPO phase ---")

    print("--- Loading and Preprocessing Data for Final Run ---")
    data = load_and_preprocess_data(for_transformer=True)
    if not data:
        print("Data loading failed. Aborting.")
        return
    X_train, y_train, X_val, y_val, X_test, y_test, _, num_features, num_classes, class_names = data

    print("--- Final Model Training ---")
    final_model, history = train_model(X_train, y_train, X_val, y_val,
                                       num_features, num_classes, class_names,
                                       strategy=strategy, trial=None)

    print("--- Final Model Evaluation ---")
    if final_model:
        evaluate_model(final_model, X_test, y_test, history, class_names)
    else:
        print("Final model is None, skipping evaluation.")

    print(f"--- Pipeline Complete at {datetime.datetime.now()} ---")

if __name__ == '__main__':
    run_on_tpu(run_hpo=True, n_hpo_trials=70)
