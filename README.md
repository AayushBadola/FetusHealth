
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/Keras-2.10%2B-red?style=for-the-badge&logo=keras" alt="Keras Version">
  <img src="https://img.shields.io/badge/Optuna-3.0%2B-8D53A0?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjODk1M0EwIiByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+T3B0dW5hPC90aXRsZT48cGF0aCBkPSJNMTIgMEM1LjM3MyAwIDAgNS4zNzMgMCAxMnM1LjM3MyAxMiAxMiAxMiAxMi01LjM3MyAxMi0xMiBTMTguNjI3IDAgMTIgMHptMCAyMS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptMy42LTE4djEuOGgtMS44djMuNmgtMy42VjUuNGgtMS44VjMuNmg3LjJ6bS0zLjYgOS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptLTMuNi0zLjZhMS4yIDEuMiAwIDEgMSAwIDIuNCAxLjIgMS4yIDAgMCAxIDAtMi40em03LjIgMGEyLjQgMi40IDAgMSAxIDAgNC44IDIuNCAyLjQgMCAwIDEgMC00Ljh6Ii8+PC9zdmc+" alt="Optuna Version">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge" alt="Code Style: Black">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

# Advanced Fetal Health Classification using Hybrid Deep Learning

This project presents a sophisticated deep learning pipeline for the classification of fetal health status based on Cardiotocography (CTG) features. The core of the project is a custom-designed hybrid neural network, combining Convolutional Neural Networks (CNNs) with Transformer Encoder mechanisms. This architecture was systematically optimized via an extensive automated hyperparameter search using Optuna, culminating in a high-performance model.

## <ins>Key Achievements & Project Status</ins>

*   **High Performance:** The final optimized `CNN-Transformer Hybrid` model achieved a **Validation Accuracy of ~95%**.
*   **Robust Pipeline:** Successfully developed and validated an end-to-end pipeline encompassing data preprocessing, custom model implementation, comprehensive hyperparameter optimization, training, and detailed evaluation.
*   **Advanced Architecture:** Leveraged a state-of-the-art hybrid model structure tailored for sequential data analysis.
*   **Status:** Model Training and Hyperparameter Optimization **Completed**.
*   **IDE Tested:** Google Colab (with T4 GPU), VSCode.

## Table of Contents

1.  [Objective](#objective)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Model Architecture: CNN-Transformer Hybrid](#model-architecture-cnn-transformer-hybrid)
    *   [Custom Keras Layers](#custom-keras-layers)
    *   [Hyperparameter Optimization with Optuna](#hyperparameter-optimization-with-optuna)
    *   [Training Regimen](#training-regimen)
    *   [Evaluation Metrics](#evaluation-metrics)
4.  [Technical Stack](#technical-stack)
5.  [Results Overview](#results-overview)
6.  [Project Structure](#project-structure)
7.  [Setup and Usage](#setup-and-usage)
8.  [Future Work](#future-work)
9.  [Author & Contact](#author--contact)

---

## 1. Objective
To develop a high-accuracy, robust deep learning model for classifying fetal health into three categories (Normal, Suspect, Pathological) using features derived from Cardiotocography (CTG) signals. The project aimed to explore advanced neural architectures and automated optimization techniques to achieve state-of-the-art predictive performance on this task.

---

## 2. Dataset
The model is trained and evaluated on the **"Fetal Health Classification"** dataset.
*   **Source:** [Kaggle - Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
*   **Features:** 21 numerical features extracted from CTG, such as baseline fetal heart rate, accelerations, uterine contractions, histogram properties, etc.
*   **Target:** `fetal_health` - Categorical (1: Normal, 2: Suspect, 3: Pathological).

---

## 3. Methodology

<details>
<summary><strong>Click to expand Methodology Details</strong></summary>

### Data Preprocessing
1.  **Loading:** Data is loaded from the provided CSV file using Pandas.
2.  **Cleaning:** Basic handling of missing values (though the dataset is generally clean).
3.  **Encoding:** The categorical target variable (`fetal_health`) is label encoded (`0, 1, 2`) and then one-hot encoded for multi-class classification with `tf.keras.utils.to_categorical`.
4.  **Splitting:** Data is stratified split into training, validation, and test sets using `sklearn.model_selection.train_test_split` to maintain class proportion across splits.
5.  **Scaling:** Input features (21 numerical values) are standardized using `sklearn.preprocessing.StandardScaler` (fit exclusively on the training set, then applied to transform training, validation, and test sets).
6.  **Reshaping:** Scaled features are reshaped to `(batch_size, sequence_length=21, channels=1)` to be compatible with 1D CNN and subsequent Transformer layers.

### Model Architecture: CNN-Transformer Hybrid
A custom hybrid architecture (`build_cnn_transformer_hybrid`) was designed:

1.  **Input Layer:** Accepts the preprocessed sequence of 21 features.
2.  **CNN Backbone:**
    *   Initial 1D Convolutional layer with Batch Normalization, LeakyReLU activation, and Dropout.
    *   Followed by a configurable number of `cnn_residual_block`s. Each residual block consists of two 1D Conv layers, Batch Normalization, LeakyReLU, and Dropout, with a shortcut connection. This backbone extracts hierarchical local features.
    *   Optional MaxPooling1D (strides=1, padding='same') is applied between residual blocks.
3.  **Transformer Encoder Block:**
    *   The feature sequence output by the CNN backbone is projected to the Transformer's `d_model` dimension via a Dense layer.
    *   **Positional Encoding:** A custom `PositionalEmbedding` layer adds sinusoidal positional information.
    *   A stack of `EncoderLayer` instances, each containing:
        *   **Multi-Head Self-Attention:** Implemented via `GlobalSelfAttention` (using `tf.keras.layers.MultiHeadAttention`) with dropout.
        *   **Feed-Forward Network (FFN):** Two dense layers with a GELU activation and dropout.
    *   Residual connections and Layer Normalization are employed after both MHA and FFN sub-layers.
4.  **Classification Head:**
    *   Outputs from the Transformer Encoder are processed via concatenated Global Average Pooling and Global Max Pooling, followed by Layer Normalization.
    *   A final MLP head (multiple Dense layers with Batch Normalization, LeakyReLU, and high Dropout) maps representations to the three output classes using a Softmax activation.

### Custom Keras Layers
Developed in `src/custom_layers.py`:
*   `PositionalEmbedding`: Injects sequence order information.
*   `BaseAttention`, `GlobalSelfAttention`: Encapsulate Multi-Head Self-Attention.
*   `FeedForward`: Position-wise FFN for Transformer blocks.
*   `EncoderLayer`: A complete Transformer Encoder layer.
*   `TransformerEncoder`: Stacks multiple `EncoderLayer`s.
*   `cnn_residual_block`: A functional utility for creating CNN residual blocks with unique naming.
    *(All custom layers include `get_config` methods for serialization and pass the `training` argument for correct behavior of dropout/batchnorm during inference vs. training.)*

### Hyperparameter Optimization with Optuna
Extensive automated HPO was conducted using **Optuna**:
*   **Objective:** Maximize `val_accuracy`.
*   **Trials:** A significant number of trials (e.g., 50-70+) were run.
*   **Study Storage & Pruning:** SQLite backend for persistence; `MedianPruner` for efficiency.
*   **Search Space Highlights:**
    *   *Architectural:* CNN layers/filters/kernels/dropout, Transformer layers/`d_model`/heads/FFN-multiplier/dropout, Dense head units/dropout.
    *   *Training:* Learning rate, weight decay, batch size, label smoothing.

### Training Regimen
*   **Optimizer:** AdamW.
*   **Loss:** Categorical Crossentropy with label smoothing.
*   **Class Imbalance:** Addressed via `class_weight='balanced'`.
*   **Callbacks:** `ModelCheckpoint` (save best by `val_accuracy`), `EarlyStopping` (monitor `val_loss`, restore best weights), `ReduceLROnPlateau`, `TensorBoard`.
*   **Environment:** GPU-accelerated training on Google Colab (T4 GPU) with XLA compilation.

### Evaluation Metrics
*   Primary: Test Accuracy.
*   Secondary: Per-class Precision, Recall, F1-score; Macro F1-score; Confusion Matrix.

</details>

---

## 4. Technical Stack
*   **Language:** Python 3.9+
*   **Core Frameworks:** TensorFlow 2.10+, Keras (via TensorFlow)
*   **Hyperparameter Optimization:** Optuna 3.0+
*   **Data Manipulation & ML:** NumPy, Pandas, Scikit-learn
*   **Visualization:** Matplotlib, Seaborn
*   **Development Environment:** Google Colab (T4 GPU), VSCode
*   **Code Style:** Black Formatter

---

## 5. Results Overview
After comprehensive hyperparameter optimization and training of the final `CNN-Transformer Hybrid` model:
*   **Peak Validation Accuracy (during HPO):** Achieved approximately **95%**.


---

## 6. Project Structure
```
FetusHealth/
├── data/
│   └── fetal_health.csv
├── models/
│   ├── optuna_study.db
│   └── best_fetal_health_hybrid_model.keras
├── results/
│   ├── best_hyperparameters.txt
│   ├── best_optuna_params.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── hpo_trials/
│   ├── logs_hpo/
│   ├── logs_final/
│   └── optuna_visualizations/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── custom_layers.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── hyperparameter_tuner.py
│   ├── model_architectures.py
│   ├── trainer.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 7. Setup and Usage

1.  **Clone:** `git clone https://github.com/AayushBadola/FetusHealth.git && cd FetusHealth`
2.  **Environment:** Create and activate a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3.  **Dependencies:** `pip install -r requirements.txt`
4.  **Dataset:** Download `fetal_health.csv` from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) into `data/`. te data is in github as well
5.  **Configuration:**
    *   Adjust `PROJECT_ROOT_ON_DRIVE` in `src/config.py` if using Google Colab with a specific Drive path. For local runs, paths should default correctly.
    *   Set `n_hpo_trials` in `main.py`'s call to `main()` for HPO.
6.  **Execution:**
    *   **Full Pipeline (HPO & Final Training):**
        ```bash
        python main.py 
        # (Ensure main.py calls main(run_hpo=True, n_hpo_trials=DESIRED_NUMBER))
        ```
    *   **Train/Evaluate with existing HPs (skip HPO):**
        ```bash
        # (Ensure main.py calls main(run_hpo=False) and results/best_optuna_params.json exists)
        python main.py
        ```
    *   **Evaluate a pre-trained model directly (example):**
        *(Requires `models/best_fetal_health_hybrid_model.keras` and `models/fetal_health_scaler.joblib` to exist. A separate script might be cleaner for this, but `main(run_hpo=False)` and ensuring no training occurs could also work if `trainer.py` is adapted to load instead of train if model exists when `trial is None`.)*

---

## 8. Future Work
*   Implement full Keras serialization (`@keras.saving.register_keras_serializable`) for all custom layers to streamline model saving/loading and eliminate the rebuild-fallback mechanism.
*   Explore ensemble methods using top models from HPO for potential performance boosts.
*   Investigate advanced data augmentation techniques suitable for tabular/sequential medical data.
*   Deploy the final model as a web service/API for real-time predictions.
*   Conduct more extensive HPO runs with broader search spaces or different Optuna samplers.

---

## 9. Author & Contact

**Aayush Badola**

Let's connect:

<p align="left">
<a href="mailto:aayush.badola2@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/></a>&nbsp;
<a href="https://www.linkedin.com/in/aayush-badola-0a7b2b343/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>&nbsp;
<a href="https://github.com/AayushBadola" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
</p>

---

<p align="center">

<img src="https://giphy.com/gifs/pudgypenguins-happy-coffee-in-my-zone-Sl7OlpTiHi9pPPZKp4" width="480" height="480">

</p>
