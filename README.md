<p align="center">
   
 <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/Keras-2.10%2B-red?style=for-the-badge&logo=keras" alt="Keras Version">
  <img src="https://img.shields.io/badge/Optuna-3.0%2B-purple?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjODk1M0EwIiByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+T3B0dW5hPC90aXRsZT48cGF0aCBkPSJNMTIgMEM1LjM3MyAwIDAgNS4zNzMgMCAxMnM1LjM3MyAxMiAxMiAxMiAxMi01LjM3MyAxMi0xMiBTMTguNjI3IDAgMTIgMHptMCAyMS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptMy42LTE4djEuOGgtMS44djMuNmgtMy42VjUuNGgtMS44VjMuNmg3LjJ6bS0zLjYgOS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptLTMuNi0zLjZhMS4yIDEuMiAwIDEgMSAwIDIuNCAxLjIgMS4yIDAgMCAxIDAtMi40em03LjIgMGEyLjQgMi40IDAgMSAxIDAgNC44IDIuNCAyLjQgMCAwIDEgMC00Ljh6Ii8+PC9zdmc+" alt="Optuna Version">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge" alt="Code Style: Black">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

# Advanced Fetal Health Classification using Hybrid Deep Learning

This project implements a sophisticated deep learning pipeline for the classification of fetal health status based on Cardiotocography (CTG) features. The core of the project is a custom-designed hybrid neural network, combining Convolutional Neural Networks (CNNs) with Transformer Encoder mechanisms, optimized via automated hyperparameter search using Optuna. The objective is to achieve high predictive accuracy and provide a robust framework for CTG data analysis.



## Project Status

**Current Phase:** Hyperparameter Optimization (Ongoing)
**Target Model:** `build_cnn_transformer_hybrid`
**IDE Tested:** Google Colab (with T4 GPU), VSCode

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
5.  [Project Structure](#project-structure)
6.  [Setup and Usage](#setup-and-usage)
7.  [Author & Contact](#author--contact)

## Objective

To develop a high-accuracy, robust deep learning model for classifying fetal health into three categories (Normal, Suspect, Pathological) using features derived from Cardiotocography (CTG) signals. The project aims to explore advanced neural architectures and automated optimization techniques to push the boundaries of predictive performance on this task.

## Dataset

The model is trained and evaluated on the **"Fetal Health Classification"** dataset.
*   **Source:** [Kaggle - Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
*   **Features:** 21 numerical features extracted from CTG, such as baseline fetal heart rate, accelerations, uterine contractions, histogram properties, etc.
*   **Target:** `fetal_health` - Categorical (1: Normal, 2: Suspect, 3: Pathological).

## Methodology

### Data Preprocessing

1.  **Loading:** Data is loaded from the provided CSV file.
2.  **Cleaning:** Basic handling of missing values (if any).
3.  **Encoding:** The categorical target variable (`fetal_health`) is label encoded and then one-hot encoded for multi-class classification.
4.  **Splitting:** Data is stratified split into training, validation, and test sets.
5.  **Scaling:** Input features (21 numerical values) are standardized using `sklearn.preprocessing.StandardScaler` (fit on the training set, transformed on validation and test sets).
6.  **Reshaping:** Scaled features are reshaped to `(batch_size, sequence_length, 1)` to be compatible with 1D CNN and subsequent Transformer layers, treating the 21 features as a sequence.

### Model Architecture: CNN-Transformer Hybrid

A custom hybrid architecture (`build_cnn_transformer_hybrid`) is central to this project:

1.  **Input Layer:** Accepts the preprocessed sequence of 21 features.
2.  **CNN Backbone:**
    *   Consists of an initial 1D Convolutional layer followed by multiple `cnn_residual_block`s.
    *   Each residual block typically includes two 1D Conv layers, Batch Normalization, LeakyReLU activation, and Dropout.
    *   Aims to extract hierarchical local features and patterns from the input sequence.
    *   Gentle MaxPooling (strides=1) is optionally applied between residual blocks.
3.  **Transformer Encoder Block:**
    *   The feature sequence output by the CNN backbone is projected to the Transformer's `d_model` dimension.
    *   **Positional Encoding:** A custom `PositionalEmbedding` layer adds sinusoidal positional information to the projected feature sequence, crucial for the Transformer to understand sequence order.
    *   Consists of multiple stacked `EncoderLayer` instances. Each `EncoderLayer` contains:
        *   **Multi-Head Self-Attention:** Implemented via a `GlobalSelfAttention` layer (utilizing `tf.keras.layers.MultiHeadAttention`) to weigh the importance of different features relative to each other across the entire sequence.
        *   **Feed-Forward Network (FFN):** A position-wise fully connected feed-forward network (typically two dense layers with a GELU activation in between).
    *   Both sub-layers (MHA and FFN) employ residual connections and Layer Normalization. Dropout is applied within MHA, FFN, and after positional encoding.
4.  **Classification Head:**
    *   The output sequence from the Transformer Encoder is processed by applying both Global Average Pooling and Global Max Pooling along the sequence dimension.
    *   These pooled outputs are concatenated and passed through Layer Normalization.
    *   A final Multi-Layer Perceptron (MLP) head, consisting of a series of Dense layers with Batch Normalization, LeakyReLU activation, and high Dropout, maps the learned representations to the three output classes.
    *   A Softmax activation function produces the final class probabilities.

### Custom Keras Layers

Several custom Keras layers were developed in `src/custom_layers.py` to implement the hybrid architecture:
*   `PositionalEmbedding`: Adds positional information to input sequences.
*   `BaseAttention`, `GlobalSelfAttention`: Building blocks for multi-head self-attention.
*   `FeedForward`: Position-wise feed-forward network for Transformer encoder layers.
*   `EncoderLayer`: A single Transformer encoder layer combining self-attention and FFN.
*   `TransformerEncoder`: Stacks multiple `EncoderLayer` instances.
*   `cnn_residual_block`: A functional definition for creating CNN residual blocks.
    *All custom layers are designed with considerations for potential Keras serialization (`get_config` methods implemented, `@keras.saving.register_keras_serializable` can be enabled).*

### Hyperparameter Optimization with Optuna

Automated hyperparameter optimization is performed using **Optuna**:
*   **Objective:** Maximize validation accuracy.
*   **Study Storage:** SQLite database (`models/optuna_study.db`) for persistence and resumption.
*   **Pruning:** `MedianPruner` is used to terminate unpromising trials early.
*   **Tunable Architectural Hyperparameters (examples):**
    *   CNN: Number of layers, base filter count, kernel size, dropout rates.
    *   Transformer: Number of encoder layers, `d_model`, number of attention heads, FFN multiplier, dropout rates.
    *   Dense Head: Number of units in dense layers, dropout rates.
*   **Tunable Training Hyperparameters:**
    *   Learning rate (log uniform distribution).
    *   Weight decay (for AdamW optimizer).
    *   Batch size.
    *   Label smoothing factor.

### Training Regimen

*   **Optimizer:** AdamW (Adam with decoupled weight decay).
*   **Loss Function:** Categorical Crossentropy with label smoothing.
*   **Class Imbalance:** Addressed using `class_weight='balanced'` during model fitting.
*   **Callbacks:**
    *   `ModelCheckpoint`: Saves the best model based on `val_accuracy`.
    *   `EarlyStopping`: Monitors `val_loss` to prevent overfitting and halt training if no improvement, restoring best weights. Patience is set more aggressively for HPO trials than for final model training.
    *   `ReduceLROnPlateau`: Dynamically reduces learning rate if `val_loss` stagnates.
    *   `TensorBoard`: For logging training metrics and graph visualization.
*   **GPU Utilization:** The pipeline is executed on Google Colab utilizing a T4 GPU, significantly accelerating training and HPO. XLA (Accelerated Linear Algebra) compilation is leveraged by TensorFlow for further performance gains.

### Evaluation Metrics

The performance of the trained models is assessed using:
*   Overall Test Accuracy
*   Test Loss
*   Per-class Precision, Recall, and F1-score (from `sklearn.metrics.classification_report`)
*   Macro-averaged F1-score, Precision, and Recall
*   Confusion Matrix visualization

## Technical Stack

*   **Language:** Python 3.x
*   **Core Libraries:** TensorFlow, Keras
*   **Hyperparameter Optimization:** Optuna
*   **Data Handling:** NumPy, Pandas
*   **Machine Learning Utilities:** Scikit-learn
*   **Plotting & Visualization:** Matplotlib, Seaborn, (Plotly for Optuna visualizations - currently optional)
*   **Development Environments:** VSCode (local), Google Colab (cloud with T4 GPU)

## Project Structure

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
│   ├── hpo_trials/         # Stores models from individual HPO trials
│   ├── logs_hpo/           # TensorBoard logs for HPO trials
│   ├── logs_final/         # TensorBoard logs for final model training
│   └── optuna_visualizations/ # Optuna study plots (if enabled)
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
├── main.py                 # Main script to run HPO, training, evaluation
├── requirements.txt        # Python package dependencies
└── README.md
```

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AayushBadola/FetusHealth.git
    cd FetusHealth
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Dataset:** Download the `fetal_health.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) and place it in the `data/` directory.
5.  **Configuration:**
    *   Review and adjust paths in `src/config.py` if necessary (especially if running on Colab with Google Drive mounted, update `PROJECT_ROOT_ON_DRIVE`).
    *   Modify `n_hpo_trials` in `main.py` as desired.
6.  **Run the pipeline:**
    *   To run hyperparameter optimization followed by final training and evaluation:
        ```bash
        python main.py 
        # (Ensure main.py calls main(run_hpo=True, n_hpo_trials=DESIRED_NUMBER))
        ```
    *   To skip HPO and train/evaluate with HPs from `config.py` or a saved `best_optuna_params.json`:
        ```bash
        # (Ensure main.py calls main(run_hpo=False))
        python main.py
        ```

## Author & Contact

**Aayush Badola**

Connect with me:

<a href="mailto:aayush.badola2@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/></a>
<a href="https://www.linkedin.com/in/aayush-badola-0a7b2b343/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="https://github.com/AayushBadola"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>

---

[![Made with Love and Coffee][love-coffee-badge]][love-coffee-link]

[python-badge]: https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white
[python-link]: https://www.python.org/
[tensorflow-badge]: https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white
[tensorflow-link]: https://www.tensorflow.org/
[keras-badge]: https://img.shields.io/badge/Keras_Core-0.1%2B-D00000.svg?style=for-the-badge&logo=keras&logoColor=white 
[keras-link]: https://keras.io/
[optuna-badge]: https://img.shields.io/badge/Optuna-3.0%2B-8D53A0.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjOEM1M0EwIiByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+T3B0dW5hPC90aXRsZT48cGF0aCBkPSJNMTIgMEM1LjM3MyAwIDAgNS4zNzMgMCAxMnM1LjM3MyAxMiAxMiAxMiAxMi01LjM3MyAxMi0xMiBTMTguNjI3IDAgMTIgMHptMCAyMS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptMy42LTE4djEuOGgtMS44djMuNmgtMy42VjUuNGgtMS44VjMuNmg3LjJ6bS0zLjYgOS42YTEuMiAxLjIgMCAxIDEgMCAyLjQgMS4yIDEuMiAwIDAgMSAwLTIuNHptLTMuNi0zLjZhMS4yIDEuMiAwIDEgMSAwIDIuNCAxLjIgMS4yIDAgMCAxIDAtMi40em03LjIgMGEyLjQgMi40IDAgMSAxIDAgNC44IDIuNCAyLjQgMCAwIDEgMC00Ljh6Ii8+PC9zdmc+
[optuna-link]: https://optuna.org/
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[black-link]: https://github.com/psf/black
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-link]: https://opensource.org/licenses/MIT
[love-coffee-badge]: https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20%26%20%E2%98%95%EF%B8%8F-black?style=for-the-badge



