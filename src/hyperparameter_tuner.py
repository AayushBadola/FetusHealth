import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import os
import shutil
import datetime

from src import config
from src.data_loader import load_and_preprocess_data
from src.trainer import train_model

def objective(trial, X_train, y_train, X_val, y_val, num_features, num_classes, class_names):
    print(f"DEBUG objective: Starting Optuna Trial {trial.number} at {datetime.datetime.now()}")
    val_accuracy = train_model(X_train, y_train, X_val, y_val,
                               num_features, num_classes, class_names, trial=trial)
    print(f"DEBUG objective: Optuna Trial {trial.number} finished. Returned val_accuracy: {val_accuracy} at {datetime.datetime.now()}")
    return val_accuracy

def run_hyperparameter_tuning(n_trials=50):
    print(f"--- Starting Advanced Hyperparameter Tuning with Optuna (run_hyperparameter_tuning called) at {datetime.datetime.now()} ---")
    
    data_load_result = load_and_preprocess_data(for_transformer=True)
    if data_load_result is None or data_load_result[0] is None:
        print("DEBUG run_hyperparameter_tuning: Data loading failed. Cannot proceed with HPO. Returning None.")
        return None

    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, num_features, num_classes, class_names) = data_load_result
    print(f"DEBUG run_hyperparameter_tuning: Data loaded successfully for HPO at {datetime.datetime.now()}.")

    db_file_path = config.OPTUNA_DB_PATH.replace("sqlite:///", "")

    print(f"DEBUG run_hyperparameter_tuning: Creating/Loading Optuna study: {config.OPTUNA_DB_PATH} at {datetime.datetime.now()}")
    study = optuna.create_study(
        study_name="fetal_health_cnn_transformer_optimization",
        direction="maximize",
        storage=config.OPTUNA_DB_PATH,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(1, n_trials // 5), n_warmup_steps=10, interval_steps=1)
    )
    print(f"DEBUG run_hyperparameter_tuning: Optuna study object created/loaded at {datetime.datetime.now()}.")

    func_objective = lambda trial_obj: objective(trial_obj, X_train, y_train, X_val, y_val,
                                                 num_features, num_classes, class_names)

    print(f"DEBUG run_hyperparameter_tuning: Starting study.optimize for {n_trials} trials at {datetime.datetime.now()}.")
    try:
        study.optimize(func_objective, n_trials=n_trials, timeout=None, n_jobs=1)
    except Exception as e_optimize:
        print(f"DEBUG run_hyperparameter_tuning: ERROR during study.optimize: {e_optimize}")
        print("DEBUG run_hyperparameter_tuning: HPO was interrupted or failed. Attempting to report best trial so far if any.")

    print(f"\n--- Hyperparameter Tuning Finished (message from run_hyperparameter_tuning) --- at {datetime.datetime.now()}")
    
    if not study.trials:
        print("DEBUG run_hyperparameter_tuning: No trials were completed or recorded in the study. Returning None.")
        return None

    print(f"Number of finished trials in study: {len(study.trials)}")
    
    best_trial = None
    try:
        best_trial = study.best_trial 
        print(f"DEBUG run_hyperparameter_tuning: Retrieved study.best_trial object.")
        print(f"Best trial raw object: {best_trial}")
        print(f"Best trial number: {best_trial.number}")
        print(f"Best trial value (validation accuracy): {best_trial.value:.4f}")
        print("Best hyperparameters from study.best_trial.params:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except RuntimeError as e_no_best_trial: 
        print(f"DEBUG run_hyperparameter_tuning: Optuna could not determine best_trial: {e_no_best_trial}")
        print(f"-------------------------------------------------------------------------------------------")
        print(f"DEBUG hyperparameter_tuner: 'best_trial' could not be determined due to RuntimeError. Returning None from run_hyperparameter_tuning at {datetime.datetime.now()}.")
        print(f"-------------------------------------------------------------------------------------------")
        return None

    best_params_path = os.path.join(config.RESULTS_DIR, "best_hyperparameters.txt")
    try:
        with open(best_params_path, "w") as f:
            f.write(f"Best trial validation accuracy: {best_trial.value:.4f}\n")
            f.write(f"From Optuna study: {study.study_name}, Best Trial Number: {best_trial.number}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        print(f"Best hyperparameters saved to {best_params_path}")
    except Exception as e_save_txt:
        print(f"DEBUG run_hyperparameter_tuner: Error saving best_hyperparameters.txt: {e_save_txt}")

    print(f"DEBUG hyperparameter_tuner: About to attempt Optuna visualizations at {datetime.datetime.now()}.")
    vis_save_dir = os.path.join(config.RESULTS_DIR, "optuna_visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)
    print(f"DEBUG hyperparameter_tuner: SKIPPING ALL PLOTLY VISUALIZATION GENERATION AND SAVING FOR DIAGNOSTICS.")

    print(f"-------------------------------------------------------------------------------------------")
    print(f"DEBUG hyperparameter_tuner: About to return from run_hyperparameter_tuning at {datetime.datetime.now()}.")
    if best_trial is not None and hasattr(best_trial, 'params') and best_trial.params is not None:
        print(f"DEBUG hyperparameter_tuner: Value of best_trial.params being returned: {best_trial.params}")
        print(f"DEBUG hyperparameter_tuner: Type of best_trial.params: {type(best_trial.params)}")
        print(f"-------------------------------------------------------------------------------------------")
        return best_trial.params
    else:
        print(f"DEBUG hyperparameter_tuner: 'best_trial' object or 'best_trial.params' is not available or is None. Returning None.")
        print(f"-------------------------------------------------------------------------------------------")
        return None 


if __name__ == '__main__':
    print("DEBUG hyperparameter_tuner: Running __main__ block for direct testing.")
