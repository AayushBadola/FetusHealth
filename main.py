import tensorflow as tf
import numpy as np
import os
import json
import datetime

from src.data_loader import load_and_preprocess_data
from src.trainer import train_model
from src.evaluate import evaluate_model
from src.hyperparameter_tuner import run_hyperparameter_tuning
from src import config

def update_config_with_best_hps(best_hps):
    print(f"DEBUG update_config: Inside with best_hps type: {type(best_hps)}, value: {best_hps}")
    if not isinstance(best_hps, dict):
        print("DEBUG update_config: best_hps is not a dictionary. Skipping update.")
        return
    for key, value in best_hps.items():
        config_attr_upper = key.upper()
        config_attr_direct = key
        if hasattr(config, config_attr_upper):
            print(f"  DEBUG update_config: Updating config.{config_attr_upper}: {getattr(config, config_attr_upper)} -> {value}")
            setattr(config, config_attr_upper, value)
        elif hasattr(config, config_attr_direct):
             print(f"  DEBUG update_config: Updating config.{config_attr_direct}: {getattr(config, config_attr_direct)} -> {value}")
             setattr(config, config_attr_direct, value)
        else:
            if key.startswith('num_heads_for_d_model_'):
                try:
                    actual_d_model = int(key.split('_')[-1])
                    current_config_d_model = getattr(config, 'D_MODEL', None)
                    if current_config_d_model == actual_d_model:
                        print(f"  DEBUG update_config: Updating config.NUM_HEADS (for D_MODEL={actual_d_model}): {getattr(config, 'NUM_HEADS', 'N/A')} -> {value}")
                        setattr(config, 'NUM_HEADS', value)
                except ValueError:
                    print(f"  DEBUG update_config: Could not parse d_model from conditional HP key {key}")
            else:
                print(f"  DEBUG update_config: Warning: Hyperparameter '{key}' from Optuna not found in config.py attributes.")
    print(f"DEBUG update_config: Finished update_config_with_best_hps at {datetime.datetime.now()}.")

def main(run_hpo=True, n_hpo_trials=50):
    print(f"DEBUG MAIN: main function started at {datetime.datetime.now()}. run_hpo={run_hpo}, n_hpo_trials={n_hpo_trials}")
    print("DEBUG MAIN: TensorFlow Version:", tf.__version__)
    print("DEBUG MAIN: Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print(f"DEBUG MAIN: Random seeds set using config.RANDOM_SEED = {config.RANDOM_SEED}")

    best_hps_to_use_for_final_run = None

    if run_hpo:
        print(f"DEBUG MAIN: run_hpo is True. Starting HPO phase for {n_hpo_trials} trials at {datetime.datetime.now()}.")
        returned_value_from_hpo = run_hyperparameter_tuning(n_trials=n_hpo_trials)
        
        print(f"--------------------------------------------------------------------")
        print(f"DEBUG MAIN: run_hyperparameter_tuning HAS RETURNED at {datetime.datetime.now()}.")
        print(f"DEBUG MAIN: Type of returned value from HPO: {type(returned_value_from_hpo)}")
        print(f"DEBUG MAIN: Returned value itself from HPO: {returned_value_from_hpo}")
        print(f"DEBUG MAIN: Is returned value from HPO None? {returned_value_from_hpo is None}")
        print(f"--------------------------------------------------------------------")

        if returned_value_from_hpo is not None and isinstance(returned_value_from_hpo, dict):
            best_hps_to_use_for_final_run = returned_value_from_hpo
            print(f"DEBUG MAIN: Assigned HPO results to best_hps_to_use_for_final_run.")
            update_config_with_best_hps(best_hps_to_use_for_final_run)
            json_path = os.path.join(config.RESULTS_DIR, "best_optuna_params.json")
            print(f"DEBUG MAIN: Attempting to write best_optuna_params.json to: {json_path}")
            try:
                with open(json_path, "w") as f_json:
                    json.dump(best_hps_to_use_for_final_run, f_json, indent=4)
                print(f"DEBUG MAIN: best_optuna_params.json written successfully.")
                print("Best hyperparameters from Optuna saved and applied to current run.")
            except Exception as e:
                print(f"DEBUG MAIN: ERROR writing best_optuna_params.json: {e}")
        elif returned_value_from_hpo is None:
            print(f"DEBUG MAIN: HPO returned None. No HPs to process.")
        else:
            print(f"DEBUG MAIN: HPO returned an unexpected type: {type(returned_value_from_hpo)}.")
            
    else: 
        print(f"DEBUG MAIN: run_hpo is False. Attempting to load HPs from file.")
        try:
            json_path = os.path.join(config.RESULTS_DIR, "best_optuna_params.json")
            print(f"DEBUG MAIN: Looking for saved HPs at: {json_path}")
            with open(json_path, "r") as f_json:
                loaded_hps_from_json = json.load(f_json)
            print(f"DEBUG MAIN: Loaded HPs from JSON: {loaded_hps_from_json}.")
            best_hps_to_use_for_final_run = loaded_hps_from_json
            update_config_with_best_hps(best_hps_to_use_for_final_run)
        except FileNotFoundError:
            print(f"DEBUG MAIN: No saved HPs JSON ({json_path}) found. Using defaults from config.py.")
        except Exception as e_json:
            print(f"DEBUG MAIN: Error loading HPs from JSON ({json_path}): {e_json}. Using defaults.")

    print(f"--------------------------------------------------------------------")
    print(f"DEBUG MAIN: Successfully passed HPO logic block.")
    print(f"DEBUG MAIN: HPs to be used for final run are: {best_hps_to_use_for_final_run}")
    print(f"--------------------------------------------------------------------")

    print("\n--- Loading and Preprocessing Data (Final Run) ---")
    print(f"DEBUG MAIN: About to call load_and_preprocess_data().")
    
    data_load_result = None
    try:
        data_load_result = load_and_preprocess_data(for_transformer=True)
        print(f"DEBUG MAIN: load_and_preprocess_data() call finished.")
        if data_load_result is None or data_load_result[0] is None:
             print("DEBUG MAIN: load_and_preprocess_data returned invalid. Exiting main.")
             return
    except Exception as e_data:
        print(f"DEBUG MAIN: ERROR during load_and_preprocess_data() call: {e_data}")
        return

    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, num_features, num_classes, class_names) = data_load_result
    print(f"DEBUG MAIN: Data unpacked. X_train shape: {X_train.shape if X_train is not None else 'None'}")

    input_shape_for_reload = (num_features, 1)
    print(f"DEBUG MAIN: input_shape_for_reload set to: {input_shape_for_reload}")

    print("\n--- Final Model Training (with best or default HPs) ---")
    print(f"DEBUG MAIN: About to call train_model for final training. trial=None will be passed.")
    
    final_model, history = None, None
    try:
        final_model, history = train_model(X_train, y_train, X_val, y_val,
                                           num_features, num_classes, class_names, trial=None)
        print(f"DEBUG MAIN: train_model for final training finished. Model is None: {final_model is None}, History is None: {history is None}")
    except Exception as e_train:
        print(f"DEBUG MAIN: ERROR during final model training (train_model call): {e_train}")

    print("\n--- Final Model Evaluation ---")
    print(f"DEBUG MAIN: About to call evaluate_model.")
    try:
        if final_model:
            evaluate_model(final_model, X_test, y_test, history, class_names,
                           input_shape_for_reload=input_shape_for_reload,
                           num_classes_for_reload=num_classes)
        else:
            print("DEBUG MAIN: Final model is None. Attempting to evaluate from saved path...")
            evaluate_model(None, X_test, y_test, None, class_names,
                           input_shape_for_reload=input_shape_for_reload,
                           num_classes_for_reload=num_classes)
        print(f"DEBUG MAIN: evaluate_model finished.")
    except Exception as e_eval:
        print(f"DEBUG MAIN: ERROR during final model evaluation (evaluate_model call): {e_eval}")

    print(f"\n--- Advanced Fetal Health Prediction Process Complete at {datetime.datetime.now()} ---")

if __name__ == '__main__':
    print(f"DEBUG __main__: Script execution started at {datetime.datetime.now()}.")
    main(run_hpo=True, n_hpo_trials=55)
    print(f"DEBUG __main__: Script execution finished at {datetime.datetime.now()}.")

