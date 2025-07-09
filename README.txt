1. PROJECT OVERVIEW
This project implements deep learning models to predict the Remaining Useful Life (RUL) of turbofan engines using NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset. The primary goal is to accurately predict when an engine will fail to enable predictive maintenance. The project explores both standalone LSTM models and Domain Adversarial Neural Networks (DANN) for cross-domain transfer learning.

2. DATASET DESCRIPTION
The NASA CMAPSS dataset consists of simulated engine run-to-failure data. It includes four sub-datasets (FD001-FD004) with different operating conditions and fault modes:

FD001: Single operating condition, single failure mode
FD002: Six operating conditions, single failure mode
FD003: Single operating condition, two failure modes
FD004: Six operating conditions, two failure modes
Each dataset contains multivariate time series data from multiple engines. Each engine starts with different degrees of initial wear and manufacturing variation and develops a fault at some point. The system is simulated until failure.

Data Format:

Unit number: Unique ID for each engine
Time cycles: Number of operational cycles
Operational settings: 3 operational settings
Sensor measurements: 21 sensor readings (temperature, pressure, etc.)




3. DIRECTORY STRUCTURE


/
├── checkpoints/               # Model checkpoints
│   ├── best_FD001_target.pt   # Target-only model for FD001
│   ├── best_FD002_target.pt   # Target-only model for FD002
│   ├── best_FD003_target.pt   # Target-only model for FD003
│   ├── best_FD004_target.pt   # Target-only model for FD004
│   ├── best_FD004_baseline_bn.pt  # Baseline model trained on FD004
│   ├── best_FD004_to_FD001_dann.pt # Domain adaptation from FD004 to FD001
│   ├── best_FD004_to_FD002_dann.pt # Domain adaptation from FD004 to FD002
│   ├── best_FD004_to_FD003_dann.pt # Domain adaptation from FD004 to FD003
│   ├── best_FD004_to_FD004_dann.pt # Domain adaptation within FD004
│   ├── scaler_FD001.bin       # Feature scalers for each dataset
│   ├── scaler_FD002.bin
│   ├── scaler_FD003.bin
│   └── scaler_FD004.bin
├── data/                      # Raw data files
│   ├── train_FD001.txt        # Training data for FD001
│   ├── train_FD002.txt        # Training data for FD002
│   ├── train_FD003.txt        # Training data for FD003
│   ├── train_FD004.txt        # Training data for FD004
│   ├── test_FD001.txt         # Test data for FD001
│   ├── test_FD002.txt         # Test data for FD002
│   ├── test_FD003.txt         # Test data for FD003
│   ├── test_FD004.txt         # Test data for FD004
│   ├── RUL_FD001.txt          # Ground truth RUL values for test data
│   ├── RUL_FD002.txt          # Ground truth RUL values for test data
│   ├── RUL_FD003.txt          # Ground truth RUL values for test data
│   ├── RUL_FD004.txt          # Ground truth RUL values for test data
│   ├── readme.txt             # Original dataset documentation
│   └── Damage Propagation Modeling.pdf  # Reference paper
├── notebooks/                 # Jupyter notebooks
│   ├── RUL_Prediction.ipynb   # Main implementation notebook
│   
└── Results/                   # Output results and visualizations



4. USAGE INSTRUCTIONS


- install requirments

pip install -r requirements.txt




- Training Baseline

from notebooks.RUL_Prediction import train_target_domain
checkpoint_path = train_target_domain('FD001')


- Evaluation Baseline


from notebooks.RUL_Prediction import evaluate_target_domain
results = evaluate_target_domain('FD001', 'checkpoints/best_FD001_target.pt')



- Training DANN Model

from notebooks.RUL_Prediction import train_dann
model_path = train_dann(source_domain='FD004', target_domain='FD001')



- Model Deployment with FastAPI

# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Open
http://127.0.0.1:8000/docs



# POST Predictions

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
  "series": [
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8},
    {"op_1":0.5,"op_2":0.3,"op_3":0.7,"s1":0.1,"s2":0.4,"s3":0.6,"s4":0.2,"s5":0.8,"s6":0.3,"s7":0.5,"s8":0.9,"s9":0.4,"s10":0.7,"s11":0.2,"s12":0.6,"s13":0.8,"s14":0.3,"s15":0.5,"s16":0.7,"s17":0.4,"s18":0.6,"s19":0.2,"s20":0.5,"s21":0.8}
  ]
}'



5. DEPENDENCIES
Python 3.8+
PyTorch 1.9+
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
FastAPI (for deployment)
Uvicorn (for deployment)
Joblib





6. REFERENCES

i. A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
ii. NASA Prognostics Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/