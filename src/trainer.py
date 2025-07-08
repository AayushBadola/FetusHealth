# FILE: src/trainer.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import datetime

from src import config
from src.model_architectures import build_cnn_transformer_hybrid
from src.utils import focal_loss

# MODIFICATION: Add 'strategy' to the function signature
def train_model(X_train, y_train, X_val, y_val, num_features, num_classes, class_names, strategy, trial=None):
    input_shape = (num_features, 1)
    print(f"DEBUG trainer.py: train_model called. Num Replicas: {strategy.num_replicas_in_sync}")

    if trial:
        print(f"DEBUG trainer.py: HPO Trial {trial.number} - Suggesting HPs.")
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        label_smoothing_val = trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.01)
        batch_size_val = trial.suggest_categorical('batch_size', [64, 128, 256]) # Larger batches for TPU
        optimizer_choice = trial.suggest_categorical('optimizer', ['AdamW'])
    else:
        print("DEBUG trainer.py: Final Training - Using HPs from config.")
        lr = getattr(config, 'LEARNING_RATE', config.INITIAL_LEARNING_RATE)
        wd = getattr(config, 'WEIGHT_DECAY', 1e-5)
        label_smoothing_val = getattr(config, 'LABEL_SMOOTHING', 0.05)
        batch_size_val = getattr(config, 'BATCH_SIZE', 128)
        optimizer_choice = 'AdamW'

    # NEW: Calculate the global batch size for distribution
    global_batch_size = batch_size_val * strategy.num_replicas_in_sync
    print(f"DEBUG trainer.py: Per-replica batch size: {batch_size_val}, Global batch size: {global_batch_size}")

    # MODIFICATION: All model and optimizer creation must be within the strategy scope.
    with strategy.scope():
        print(f"DEBUG trainer.py: Building model inside strategy.scope()")
        # Your model architecture call is UNCHANGED
        model = build_cnn_transformer_hybrid(input_shape, num_classes, hp=trial)

        optimizer = AdamW(learning_rate=lr, weight_decay=wd, clipnorm=config.GRADIENT_CLIP_NORM)

        # MODIFICATION: Use Reduction.NONE for loss function with TPUStrategy.
        # The strategy handles the reduction across replicas automatically.
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing_val,
            reduction=tf.keras.losses.Reduction.NONE
        )
        print(f"DEBUG trainer.py: Using CategoricalCrossentropy with Reduction.NONE for TPU")

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.F1Score(average='macro', name='f1_macro', dtype=tf.float32)
                              ])
    # END of strategy.scope()

    if not trial:
        print("DEBUG trainer.py: Model Summary for Final Training (CNN-Transformer):")
        model.summary(line_length=150)

    model_save_path = config.MODEL_SAVE_PATH
    if trial:
        model_save_path = os.path.join(config.RESULTS_DIR, "hpo_trials", f"trial_{trial.number}_best_model.keras")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1 if not trial else 0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config.REDUCE_LR_FACTOR, patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LEARNING_RATE, verbose=1)
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    y_train_indices = np.argmax(y_train, axis=1)
    class_weights_dict = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)))
    if not trial:
        print(f"DEBUG trainer.py: Using class weights: {class_weights_dict}")

    print(f"\n--- Starting Training (Trial: {trial.number if trial else 'Final'}) ---")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        # MODIFICATION: Use the global batch size here
        batch_size=global_batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weights_dict,
        verbose=1
    )
    print(f"--- Training Finished (Trial: {trial.number if trial else 'Final'}) ---")

    if trial:
        val_accuracy_trial = max(history.history.get('val_accuracy', [0]))
        print(f"DEBUG trainer.py: Trial {trial.number} finished with best val_accuracy: {val_accuracy_trial:.4f}")
        return val_accuracy_trial

    print(f"DEBUG trainer.py: Final training complete. Returning model and history.")
    return model, history
