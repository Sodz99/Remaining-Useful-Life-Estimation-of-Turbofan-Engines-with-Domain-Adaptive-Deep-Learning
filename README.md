# Remaining Useful Life Estimation of Turbofan Engines with Domain-Adaptive Deep Learning

This repository contains the code and resources for the research project on predicting the Remaining Useful Life (RUL) of turbofan engines using domain-adaptive deep learning, as detailed in the accompanying report, `Sohan_Arun_AML_Report.pdf`.

## Project Overview

The primary goal of this project is to develop a robust predictive maintenance framework for aircraft turbofan engines. By accurately forecasting the RUL, we can enable proactive maintenance scheduling, which enhances operational safety, increases availability, and reduces costs associated with unscheduled downtime.

A key challenge in real-world prognostics is **domain shift**, where a model's performance degrades because the distribution of data from the deployment environment differs from the training environment. This project directly addresses this challenge by implementing a **Domain-Adversarial Neural Network (DANN)** to learn domain-invariant features, allowing the model to generalize effectively across different operating conditions and fault modes.

## Key Achievements

- **High-Accuracy RUL Prediction:** Developed a deep learning pipeline capable of accurately predicting RUL on the benchmark NASA CMAPSS dataset.
- **Cross-Domain Generalization:** Implemented a DANN-LSTM that successfully mitigates domain shift, achieving over a **30% reduction in Root Mean Squared Error (RMSE)** compared to a baseline LSTM when transferring between dissimilar operational regimes.
- **Model Interpretability:** Utilized SHAP (SHapley Additive exPlanations) to provide clear insights into the model's decision-making process, confirming that the model learned physically meaningful relationships from the sensor data.

## Methodology

The project follows the CRISP-DM methodology, encompassing data understanding, preparation, modeling, and evaluation.

1.  **Feature Engineering:** A multi-stage pipeline prepares the data for modeling. This includes:
    - Removing static sensor channels with near-zero variance.
    - Applying a moving median filter to mitigate sensor noise.
    - Capping the RUL at 125 cycles to create a piecewise linear target, focusing the model on the most critical non-linear degradation phase.
    - Performing correlation-based feature selection to isolate the most informative sensors.
    - Segmenting the time-series data into windows of 30 time steps for LSTM processing.

2.  **Modeling (DANN-LSTM):**
    - A **Domain-Adversarial Neural Network (DANN)** with a two-layer bidirectional LSTM (Bi-LSTM) was architected in PyTorch.
    - The model uses a **Gradient Reversal Layer (GRL)** to adversarially train a domain classifier against the feature extractor. This forces the model to learn features that are predictive of RUL but not indicative of the source domain, leading to better generalization.

3.  **Evaluation:**
    - Model performance was quantitatively evaluated using **RMSE** and the **NASA Scoring Function**, which asymmetrically penalizes late RUL predictions to better reflect real-world maintenance costs.

## Repository Structure

```
.
├── checkpoints/         # Saved model weights for different training runs
├── data/               # NASA CMAPSS dataset files
├── notebooks/
│   └── RUL_Prediction.ipynb  # Main Jupyter Notebook with the complete workflow
├── Results/              # Output plots, including SHAP analysis and result comparisons
├── Sohan_Arun_AML_Report.pdf # The detailed research paper for this project
├── requirements.txt      # A comprehensive list of Python dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

This project is built with Python 3. The key libraries required to run the analysis and modeling notebook are:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `torch`
-   `matplotlib`
-   `seaborn`

A full list of packages used in the development environment is available in `requirements.txt`.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Sodz99/Remaining-Useful-Life-Estimation-of-Turbofan-Engines-with-Domain-Adaptive-Deep-Learning.git
    cd Remaining-Useful-Life-Estimation-of-Turbofan-Engines-with-Domain-Adaptive-Deep-Learning
    ```

2.  Install the required packages. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The entire project workflow, from data loading and preprocessing to model training and evaluation, is contained in the Jupyter Notebook:
**`notebooks/RUL_Prediction.ipynb`**

Open and run the cells in this notebook to reproduce the results.

## Citation

For a detailed understanding of the methodology and results, please refer to the research paper included in this repository:

**`Sohan_Arun_AML_Report.pdf`**
