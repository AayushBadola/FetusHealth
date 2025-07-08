# FILE: src/hyperparameter_tuner.py

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import os
import shutil
import datetime

from src import config
from src.data_loader import load_and_preprocess_data
from src.trainer import train_model

# MODIFICATION: The objective function now accepts the 'strategy' object
def objective(trial, X_train, y_train, X_val, y_val, num_features, num_classes, class_names, strategy):
    print(f"DEBUG objective: Starting Optuna Trial {trial.number} at {datetime.datetime.now()}")
    # MODIFICATION: Pass the strategy down to train_model
    val_accuracy = train_model(X_train, y_train, X_val, y_val,
                               num_features, num_classes, class_names, strategy, trial=trial)
    print(f"DEBUG objective: Optuna Trial {trial.number} finished. Returned val_accuracy: {val_accuracy} at {datetime.datetime.now()}")
    return val_accuracy

# MODIFICATION: The main tuning function now accepts the 'strategy' object
def run_hyperparameter_tuning(n_trials, strategy):
    print(f"--- Starting HPO with Optuna on {strategy.num_replicas_in_sync} TPU cores ---")
    data_load_result = load_and_preprocess_data(for_transformer=True)
    if data_load_result is None or data_load_result[0] is None:
        print("DEBUG run_hyperparameter_tuning: Data loading failed. Aborting HPO.")
        return None

    (X_train, y_train, X_val, y_val, _, _, _, num_features, num_classes, class_names) = data_load_result
    print(f"DEBUG run_hyperparameter_tuning: Data loaded successfully for HPO.")

    study = optuna.create_study(
        study_name="fetal_health_tpu_optimization",
        direction="maximize",
        storage=config.OPTUNA_DB_PATH,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1, n_trials // 5), n_warmup_steps=10, interval_steps=1)
    )

    # MODIFICATION: Use a lambda function to pass the extra 'strategy' argument to the objective
    func_objective = lambda trial_obj: objective(trial_obj, X_train, y_train, X_val, y_val,
                                                 num_features, num_classes, class_names, strategy)

    print(f"DEBUG run_hyperparameter_tuning: Starting study.optimize for {n_trials} trials with n_jobs=1 (Required for TPU).")
    try:
        # MODIFICATION: n_jobs MUST be 1 when using TPUs.
        study.optimize(func_objective, n_trials=n_trials, n_jobs=1)
    except Exception as e_optimize:
        print(f"DEBUG run_hyperparameter_tuning: ERROR during study.optimize: {e_optimize}")

    print(f"\n--- Hyperparameter Tuning Finished ---")
    if not study.trials:
        print("DEBUG run_hyperparameter_tuning: No trials completed. Returning None.")
        return None

    try:
        best_trial = study.best_trial
        print(f"Best trial number: {best_trial.number}")
        print(f"Best trial value (validation accuracy): {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        return best_trial.params
    except Exception as e:
        print(f"Could not retrieve best trial: {e}. Returning None.")
        return None
