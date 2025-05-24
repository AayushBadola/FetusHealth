# src/trainer.py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import datetime

from src import config
from src.model_architectures import build_simple_1d_cnn, build_cnn_transformer_hybrid
from src.utils import focal_loss

def train_model(X_train, y_train, X_val, y_val, num_features, num_classes, class_names, trial=None):
    input_shape = (num_features, 1)
    print(f"DEBUG trainer.py: train_model called. Input_shape={input_shape}, num_classes={num_classes}, trial_is_None={trial is None}")

    if trial:
        print(f"DEBUG trainer.py: HPO Trial {trial.number} - Suggesting HPs for build_cnn_transformer_hybrid.")
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        label_smoothing_val = trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.01)
        batch_size_val = trial.suggest_categorical('batch_size', [16, 32, 64])
        optimizer_choice = trial.suggest_categorical('optimizer', ['AdamW'])
    else:
        print("DEBUG trainer.py: Final Training - Using HPs from config for build_cnn_transformer_hybrid.")
        lr = getattr(config, 'LEARNING_RATE', getattr(config, 'INITIAL_LEARNING_RATE', 1e-4))
        wd = getattr(config, 'WEIGHT_DECAY', 1e-5)
        label_smoothing_val = getattr(config, 'LABEL_SMOOTHING', 0.05)
        batch_size_val = getattr(config, 'BATCH_SIZE', 32)
        optimizer_choice = 'AdamW'

    print(f"DEBUG trainer.py: Building model using build_cnn_transformer_hybrid. LR={lr}, WD={wd}, LabelSmoothing={label_smoothing_val}, BatchSize={batch_size_val}")
    model = build_cnn_transformer_hybrid(input_shape, num_classes, hp=trial)
    print(f"DEBUG trainer.py: Using build_cnn_transformer_hybrid. Model type: {type(model)}")

    if optimizer_choice == 'AdamW':
        optimizer = AdamW(learning_rate=lr, weight_decay=wd, clipnorm=getattr(config, 'GRADIENT_CLIP_NORM', 1.0))
    else:
        print(f"Warning: Optimizer choice '{optimizer_choice}' not fully handled, defaulting to AdamW setup.")
        optimizer = AdamW(learning_rate=lr, weight_decay=wd, clipnorm=getattr(config, 'GRADIENT_CLIP_NORM', 1.0))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_val)
    print(f"DEBUG trainer.py: Using CategoricalCrossentropy with label_smoothing={label_smoothing_val}")

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.F1Score(average='macro', name='f1_macro', dtype=tf.float32)
                          ])

    if not trial:
        print("DEBUG trainer.py: Model Summary for Final Training (CNN-Transformer):")
        model.summary(line_length=150)

    model_save_path = config.MODEL_SAVE_PATH
    if trial:
        model_save_path = os.path.join(config.RESULTS_DIR, "hpo_trials", f"trial_{trial.number}_best_model.keras")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    print(f"DEBUG trainer.py: ModelCheckpoint path: {model_save_path}")

    checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1 if not trial else 0
    )

    es_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 50)
    if trial:
        es_patience = max(10, es_patience // 2)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=es_patience, restore_best_weights=True, verbose=1
    )

    reduce_lr_patience_val = getattr(config, 'REDUCE_LR_PATIENCE', 15)
    if trial:
        reduce_lr_patience_val = max(5, reduce_lr_patience_val // 2)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=getattr(config, 'REDUCE_LR_FACTOR', 0.2),
        patience=reduce_lr_patience_val, min_lr=getattr(config, 'MIN_LEARNING_RATE', 1e-7), verbose=1
    )

    log_dir_base = config.RESULTS_DIR
    if trial:
        log_dir = os.path.join(log_dir_base, "logs_hpo", f"trial_{trial.number}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    else:
        log_dir = os.path.join(log_dir_base, "logs_final", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"DEBUG trainer.py: TensorBoard log_dir: {log_dir}")

    callbacks_list = [checkpoint, early_stopping, reduce_lr]
    if not trial or (trial.number % getattr(config, 'TENSORBOARD_HPO_TRIAL_INTERVAL', 10) == 0):
        callbacks_list.append(tensorboard_callback)

    y_train_indices = np.argmax(y_train, axis=1)
    class_weights_dict = None
    try:
        class_weights_calculated = compute_class_weight(
            'balanced', classes=np.unique(y_train_indices), y=y_train_indices
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights_calculated)}
        if not trial:
            print(f"DEBUG trainer.py: Using class weights: {class_weights_dict}")
    except Exception as e_cw:
        print(f"DEBUG trainer.py: Error calculating class weights: {e_cw}. Proceeding without class weights.")
        class_weights_dict = None

    print(f"\n--- Starting Training (Trial: {trial.number if trial else 'Final'}) ---")
    history = model.fit(
        X_train, y_train,
        epochs=getattr(config, 'EPOCHS', 300),
        batch_size=batch_size_val,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1 if not trial else 0
    )
    print(f"--- Training Finished (Trial: {trial.number if trial else 'Final'}) ---")

    if trial:
        print(f"DEBUG trainer.py: HPO Trial {trial.number} finished training. Evaluating for Optuna.")
        try:
            if os.path.exists(model_save_path):
                model.load_weights(model_save_path)
                print(f"DEBUG trainer.py: Loaded best weights for HPO trial {trial.number} from {model_save_path}")
            val_loss_trial, val_accuracy_trial, _, _, _ = model.evaluate(X_val, y_val, verbose=0)
            print(f"DEBUG trainer.py: Trial {trial.number} re-evaluated best model: Val Acc: {val_accuracy_trial:.4f}")
            return val_accuracy_trial
        except Exception as e_trial_eval:
            print(f"DEBUG trainer.py: Could not reliably get val_accuracy for trial {trial.number}: {e_trial_eval}")
            if 'val_accuracy' in history.history and history.history['val_accuracy']:
                max_hist_val_acc = np.max(history.history['val_accuracy'])
                print(f"DEBUG trainer.py: Falling back to max val_accuracy from history for trial {trial.number}: {max_hist_val_acc:.4f}")
                return max_hist_val_acc
            print(f"DEBUG trainer.py: No val_accuracy found for trial {trial.number}, returning 0.0")
            return 0.0

    print(f"DEBUG trainer.py: Final training complete. Loading best model from: {config.MODEL_SAVE_PATH}")

    best_model = None
    try:
        best_model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
        print(f"DEBUG trainer.py: Successfully loaded final best model via tf.keras.models.load_model from {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning (trainer.py): Could not load final best model directly from {config.MODEL_SAVE_PATH}. Error: {e}")
        print("This is expected if custom layers are not registered. Falling back to rebuilding and loading weights.")

        print(f"DEBUG trainer.py: Rebuilding model (cnn_transformer_hybrid) with input_shape={input_shape}, num_classes={num_classes}, using config HPs.")
        best_model = build_cnn_transformer_hybrid(input_shape, num_classes, hp=None)

        print(f"DEBUG trainer.py: Loading weights from {config.MODEL_SAVE_PATH} into rebuilt cnn_transformer_hybrid structure.")
        try:
            best_model.load_weights(config.MODEL_SAVE_PATH)
            print("DEBUG trainer.py: Model weights loaded into rebuilt cnn_transformer_hybrid structure.")
        except Exception as e_load_weights:
            print(f"DEBUG trainer.py: ERROR loading weights into rebuilt cnn_transformer_hybrid structure: {e_load_weights}")
            return best_model, history 

        print("DEBUG trainer.py: Model weights loaded. NOW RE-COMPILING THE REBUILT cnn_transformer_hybrid MODEL...")

        final_train_lr = lr 
        final_train_wd = wd
        final_train_label_smoothing = label_smoothing_val

        print(f"DEBUG trainer.py: Re-compiling rebuilt cnn_transformer_hybrid model with LR={final_train_lr}, WD={final_train_wd}, LabelSmoothing={final_train_label_smoothing}")

        optimizer_recompile = AdamW(learning_rate=final_train_lr, weight_decay=final_train_wd, clipnorm=getattr(config, 'GRADIENT_CLIP_NORM', 1.0))
        loss_fn_recompile = tf.keras.losses.CategoricalCrossentropy(label_smoothing=final_train_label_smoothing)

        best_model.compile(optimizer=optimizer_recompile,
                           loss=loss_fn_recompile,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.F1Score(average='macro', name='f1_macro', dtype=tf.float32)
                                   ])
        print("DEBUG trainer.py: Final best model (cnn_transformer_hybrid) RE-COMPILED successfully after rebuilding.")

    if not isinstance(best_model, tf.keras.Model):
        print("DEBUG trainer.py: 'best_model' is not a Keras Model instance after loading attempts.")

    print(f"DEBUG trainer.py: Returning from train_model. best_model is None: {best_model is None}")
    return best_model, history
