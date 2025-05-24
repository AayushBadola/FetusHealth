import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score as sklearn_f1_score
import os

from src import config
from src.model_architectures import build_cnn_transformer_hybrid

def plot_training_history(history, save_dir):
    print(f"DEBUG plot_training_history: Attempting to plot history. History keys: {history.history.keys()}")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Accuracy data missing', ha='center'); plt.title('Model Accuracy (Data Missing)')

    plt.subplot(1, 3, 2)
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Loss data missing', ha='center'); plt.title('Model Loss (Data Missing)')

    plt.subplot(1, 3, 3)
    f1_key, val_f1_key = None, None
    if 'f1_macro' in history.history and 'val_f1_macro' in history.history:
        f1_key, val_f1_key = 'f1_macro', 'val_f1_macro'
    elif 'f1_score' in history.history and 'val_f1_score' in history.history:
        f1_key, val_f1_key = 'f1_score', 'val_f1_score'
    
    if f1_key and val_f1_key:
        plt.plot(history.history[f1_key], label=f'Train {f1_key.replace("_", " ").title()}')
        plt.plot(history.history[val_f1_key], label=f'Validation {val_f1_key.replace("_", " ").title()}')
        plt.title(f'Model {f1_key.replace("_", " ").title()}'); plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'F1 score data missing', ha='center'); plt.title('Model F1 Score (Data Missing)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    try:
        plt.savefig(save_path)
        print(f"DEBUG plot_training_history: Training history plot saved to {save_path}")
    except Exception as e_plot_save:
        print(f"DEBUG plot_training_history: Error saving training history plot: {e_plot_save}")
    plt.close()

def plot_confusion_matrix(y_true_labels, y_pred_labels, class_names, save_dir):
    print(f"DEBUG plot_confusion_matrix: Generating confusion matrix. Class names: {class_names}")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        save_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        print(f"DEBUG plot_confusion_matrix: Confusion matrix plot saved to {save_path}")
    except Exception as e_cm_plot:
        print(f"DEBUG plot_confusion_matrix: Error generating or saving confusion matrix plot: {e_cm_plot}")
    finally:
        plt.close()

def evaluate_model(model, X_test, y_test, history, class_names, input_shape_for_reload=None, num_classes_for_reload=None):
    print(f"\n--- Evaluating on Test Set (evaluate_model called) --- History is None: {history is None}")
    
    if model is None:
        print(f"DEBUG evaluate.py: Model is None. Trying to load from {config.MODEL_SAVE_PATH}")
        try:
            loaded_model_temp = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
            print(f"DEBUG evaluate.py: Successfully loaded model via tf.keras.models.load_model from {config.MODEL_SAVE_PATH}")
            model = loaded_model_temp
        except Exception as e:
            print(f"Warning (evaluate.py): Could not load model directly from {config.MODEL_SAVE_PATH}. Error: {e}")
            print("This is expected if using complex custom layers not registered. Falling back to rebuilding.")
            
            if input_shape_for_reload and num_classes_for_reload:
                print("DEBUG evaluate.py: Rebuilding model structure (using cnn_transformer_hybrid)...")
                rebuilt_model = build_cnn_transformer_hybrid(input_shape_for_reload, num_classes_for_reload, hp=None)
                
                print(f"DEBUG evaluate.py: Loading weights from {config.MODEL_SAVE_PATH} into rebuilt structure...")
                try:
                    rebuilt_model.load_weights(config.MODEL_SAVE_PATH)
                    print("DEBUG evaluate.py: Model weights loaded into rebuilt structure.")
                except Exception as e_load_weights:
                    print(f"DEBUG evaluate.py: ERROR loading weights into rebuilt structure: {e_load_weights}")
                    return None, None

                print("DEBUG evaluate.py: Rebuilt model weights loaded. NOW RE-COMPILING THE MODEL...")
                current_lr = getattr(config, 'LEARNING_RATE', getattr(config, 'INITIAL_LEARNING_RATE', 1e-4))
                current_wd = getattr(config, 'WEIGHT_DECAY', 1e-5)
                current_label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.05)

                print(f"DEBUG evaluate.py: Re-compiling with LR={current_lr}, WD={current_wd}, LabelSmoothing={current_label_smoothing}")

                optimizer_recompile = tf.keras.optimizers.AdamW(learning_rate=current_lr, weight_decay=current_wd)
                loss_fn_recompile = tf.keras.losses.CategoricalCrossentropy(label_smoothing=current_label_smoothing)
                
                rebuilt_model.compile(optimizer=optimizer_recompile,
                                      loss=loss_fn_recompile,
                                      metrics=['accuracy',
                                               tf.keras.metrics.Precision(name='precision'),
                                               tf.keras.metrics.Recall(name='recall'),
                                               tf.keras.metrics.F1Score(average='macro', name='f1_macro', dtype=tf.float32)
                                              ])
                print("DEBUG evaluate.py: Model RE-COMPILED successfully after rebuilding.")
                model = rebuilt_model
            else:
                print("DEBUG evaluate.py: Error: Cannot rebuild model - input_shape or num_classes not provided for reload.")
                return None, None 
    
    if model is None:
        print("DEBUG evaluate.py: Model is still None after all loading attempts. Cannot evaluate.")
        return None, None

    print("DEBUG evaluate.py: Proceeding to model.evaluate() call.")
    loss, accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        loss, accuracy, precision, recall, f1 = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Macro (from model.evaluate): {f1:.4f}")
    except Exception as eval_error:
        print(f"DEBUG evaluate.py: ERROR during model.evaluate() call: {eval_error}")
        return None, None

    print("DEBUG evaluate.py: Making predictions for classification report and confusion matrix...")
    y_pred_proba = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    print("DEBUG evaluate.py: Predictions made and labels extracted.")

    print("\nClassification Report:")
    report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4)
    print(report)
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 Macro (sklearn calculated): {sklearn_f1_score(y_true_labels, y_pred_labels, average='macro'):.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    if history:
        print(f"DEBUG evaluate.py: History object received, calling plot_training_history.")
        plot_training_history(history, config.RESULTS_DIR)
    else:
        print("DEBUG evaluate.py: No history object received, skipping plot_training_history.")
    
    print(f"DEBUG evaluate.py: Calling plot_confusion_matrix.")
    plot_confusion_matrix(y_true_labels, y_pred_labels, class_names, config.RESULTS_DIR)

    if history and 'val_accuracy' in history.history:
        best_val_accuracy = max(history.history['val_accuracy'])
        print(f"\nBest Validation Accuracy during training: {best_val_accuracy:.4f}")
        target_acc_val = getattr(config, 'TARGET_ACCURACY', 0.97)
        if accuracy >= target_acc_val:
            print(f"Congratulations! Achieved >= {target_acc_val*100}% Test Accuracy target!")
        else:
            print(f"Target of {target_acc_val*100}%+ Test Accuracy not met. Current: {accuracy*100:.2f}%")
    
    print(f"DEBUG evaluate.py: evaluate_model function finished.")
    return accuracy, loss
