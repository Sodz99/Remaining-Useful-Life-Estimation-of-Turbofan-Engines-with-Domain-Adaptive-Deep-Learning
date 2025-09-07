# Turbofan Engine RUL Prediction with Domain Adaptation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A reimplementation of [Remaining Useful Lifetime Prediction via Deep Domain Adaptation)](https://arxiv.org/abs/1907.07480) from Costa et al. (2019) for predicting when aircraft turbofan engines will fail. This project was developed as a final project for the Advanced Machine Learning course at BTH, implementing Domain-Adversarial Neural Networks (DANN) and achieving about 30% better RMSE compared to regular LSTM models.

## What This Project Does

This system predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset. The main challenge here is that models trained on one type of operating condition often don't work well when you try to use them on different conditions - this is called domain shift and it's a real problem in industrial settings.

The solution uses a DANN-LSTM architecture that learns features that are good at predicting RUL but can't tell which domain the data came from. This way, the model works much better when you transfer it between different operating conditions.

### Key Features and Results

- **Domain Adaptation**: Uses DANN-LSTM with a Gradient Reversal Layer that's pretty clever at handling domain shift
- **Better Performance**: Achieves around 30% improvement in RMSE over baseline LSTM models across different datasets
- **Cross-Domain Transfer**: The model actually works well when trained on one dataset and tested on another
- **Interpretable Results**: Includes SHAP analysis so you can see what the model is actually looking at
- **API Ready**: Features a FastAPI server so you can actually use the model in practice

## How Well It Works

The results are pretty solid - here's what the implementation achieved compared to baseline LSTM models:

- **NASA Scoring Function**: Uses the official scoring that penalizes late predictions more (since that's more costly in real maintenance)
- **Cross-Domain Testing**: Tested transfer between all four FD datasets (FD001-FD004)
- **Feature Engineering**: Multi-stage preprocessing with noise filtering that actually makes a difference
- **Model Insights**: SHAP analysis shows the model learned physically meaningful patterns, which is reassuring

## Project Structure

```
├── 📂 checkpoints/           # Saved model weights from different training runs
│   ├── best_FD001_target.pt  # Models trained on single domains
│   ├── best_FD004_to_FD001_dann.pt  # Domain adaptation models
│   └── scaler_*.bin          # Feature scaling parameters
├── 📂 data/                  # NASA CMAPSS dataset files
│   ├── train_FD001.txt       # Training data for 4 different scenarios
│   ├── test_FD001.txt        # Test data
│   └── RUL_FD001.txt         # Ground truth RUL values
├── 📂 notebooks/             
│   └── RUL_Prediction.ipynb  # Main notebook with everything
├── 📂 Results/               # Plots and analysis results
├── 📄 Sohan_Arun_AML_Report.pdf  # Detailed report with methodology
├── 📄 requirements.txt       # Python packages needed
└── 📄 README.md             # This file
```

## Getting Started

### What You Need

- Python 3.8 or newer
- A GPU is helpful for training but not required

### Setting It Up

1. **Clone and navigate**
   ```bash
   git clone https://github.com/yourusername/turbofan-rul-prediction.git
   cd turbofan-rul-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install packages**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

#### Training Models

**Regular LSTM baseline:**
```python
from notebooks.RUL_Prediction import train_target_domain
checkpoint_path = train_target_domain('FD001')
```

**Domain adaptation model:**
```python
from notebooks.RUL_Prediction import train_dann
model_path = train_dann(source_domain='FD004', target_domain='FD001')
```

#### Testing Models

```python
from notebooks.RUL_Prediction import evaluate_target_domain
results = evaluate_target_domain('FD001', 'checkpoints/best_FD001_target.pt')
```

#### Using the API

Start the server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then go to `http://localhost:8000/docs` to see the interactive docs.

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [
      {"op_1": 0.5, "op_2": 0.3, "op_3": 0.7, "s1": 0.1, ...},
      ...  # 30 timesteps of sensor readings
    ]
  }'
```

## About the Dataset

Uses the **NASA CMAPSS dataset** which has simulated turbofan engine data:

| Dataset | Operating Conditions | Failure Modes | Engines (Train/Test) |
|---------|---------------------|----------------|---------------------|
| FD001   | 1                   | 1              | 100 / 100          |
| FD002   | 6                   | 1              | 260 / 259          |
| FD003   | 1                   | 2              | 100 / 100          |
| FD004   | 6                   | 2              | 248 / 249          |

Each engine has:
- 3 operational settings (altitude, throttle, etc.)
- 21 sensor readings (temperatures, pressures, vibrations, etc.)
- Run-to-failure trajectories starting from different wear levels

## Implementation Approach

### Data Preprocessing
- **Noise filtering**: Moving median filter to smooth out sensor noise
- **Feature selection**: Picks the most informative sensors using correlation analysis
- **RUL capping**: Uses a 125-cycle threshold for the linear degradation assumption
- **Windowing**: Creates 30-timestep sequences for the LSTM to process

### Model Design
- **Bidirectional LSTM**: Looks at patterns in both time directions
- **Gradient Reversal Layer**: The clever part that makes domain adaptation work
- **Multi-task learning**: Predicts RUL while trying not to be able to tell domains apart

### Evaluation
- **RMSE**: Standard error metric
- **NASA Scoring Function**: Asymmetric penalty that matters more for real applications

## Results

| Model | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|-------|------------|------------|------------|------------|
| Baseline LSTM | 18.45 | 22.89 | 19.82 | 23.67 |
| DANN-LSTM | **12.76** | **16.23** | **13.94** | **17.45** |
| **Improvement** | **30.9%** | **29.1%** | **29.7%** | **26.3%** |

## What the Model Actually Learns

The SHAP analysis reveals some interesting insights:
- **Temperature sensors** (T24, T30, T50) are the most important features
- **Pressure ratios** correlate well with engine degradation
- The model learned patterns that make sense from an engineering perspective

For a detailed understanding of the methodology and results, check out the comprehensive research paper included in this repository: **`Sohan_Arun_AML_Report.pdf`**

## Tools and Libraries

- **Deep Learning**: PyTorch for the neural networks, scikit-learn for preprocessing
- **Data Handling**: Pandas and NumPy for data manipulation
- **Plotting**: Matplotlib and Seaborn for visualizations
- **Interpretability**: SHAP for understanding model decisions
- **Deployment**: FastAPI and Uvicorn for the web API
- **Development**: Jupyter Notebooks for the main workflow


## Credits

- **Original Research**: Li et al. "[Remaining Useful Lifetime Prediction via Deep Domain Adaptation](https://arxiv.org/abs/1907.07480)" (2019)
- **Dataset**: NASA Prognostics Center of Excellence for making CMAPSS available
- **Institution**: BTH Advanced Machine Learning course and instructors
- **Community**: PyTorch developers and the broader ML community

## 👤 Author

**Sohan Arun**  
Master’s Student, Computer Science  
Blekinge Institute of Technology, Sweden  
📧 [Sohanoffice46@gmail.com](mailto:Sohanoffice46@gmail.com)

