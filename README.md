# Fraud Detection System

A machine learning project for detecting fraudulent transactions using various ML algorithms with Weights & Biases integration for experiment tracking and model management.

## Project Overview

This project demonstrates a complete fraud detection pipeline including:
- Data preprocessing and feature engineering
- Multiple ML model implementations
- Experiment tracking with Weights & Biases
- Model evaluation and comparison
- Hyperparameter optimization

## Project Structure

```
Fraud-Detector/
├── data/                 # Raw and processed data
├── models/              # Saved models and artifacts
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── data/           # Data processing scripts
│   ├── models/         # Model definitions
│   ├── training/       # Training scripts
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## Code Overview

### Data Generation (`src/data/data_generator.py`)
- Creates synthetic fraud detection data with realistic transaction features
- Generates fraud patterns based on real-world scenarios (large amounts, late night transactions, new accounts)
- Adds engineered features (amount deviation, transaction frequency, time-based flags)
- Handles class imbalance with configurable fraud ratio (default 1%)

### Model Factory (`src/models/model_factory.py`)
- Provides multiple ML models with consistent interfaces
- Implements Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, and Neural Network
- ModelFactory class creates model instances based on type and parameters
- Each model handles feature importance and parameter logging

### W&B Utilities (`src/utils/wandb_utils.py`)
- Weights & Biases integration utilities (mix of native W&B functions and custom wrappers)
- init_experiment(): Sets up W&B runs with configuration
- log_metrics(): Logs performance metrics (precision, recall, F1, ROC-AUC)
- log_confusion_matrix(): Creates and logs confusion matrix visualizations
- save_model_artifact(): Saves trained models as W&B artifacts
- log_feature_importance(): Tracks which features are most important
- create_sweep_config(): Sets up hyperparameter optimization

### Training Pipeline (`src/training/train.py`)
- Main training pipeline that orchestrates the entire process
- FraudDetectionTrainer class manages data generation, preprocessing, model training, and evaluation
- Loads configuration from YAML files
- Applies SMOTE resampling for class balance
- Supports single model or multi-model training
- Logs all results to W&B

### Project Initialization (`src/init_wandb.py`)
- Project initialization and setup
- Creates W&B project configuration
- Generates default YAML config files for experiments
- Sets up .gitignore file
- Provides project structure setup

### Data Exploration (`notebooks/data_exploration.py`)
- Data analysis and visualization script
- Generates sample dataset and shows basic statistics
- Analyzes correlations between features and fraud
- Provides insights into fraud patterns and data distribution

### Configuration (`configs/` directory)
- YAML-based configuration management
- Contains experiment-specific config files
- Defines model parameters, data settings, training options
- Allows easy experiment configuration without code changes

## Data Flow

1. **Data Generation** → Creates synthetic fraud data with realistic patterns
2. **Data Preprocessing** → Scales features, applies resampling for class balance
3. **Model Training** → Trains multiple ML algorithms
4. **Evaluation** → Calculates fraud detection metrics
5. **Logging** → Sends all results to W&B for tracking and visualization

## Key Features

- **Modular Design**: Each component can be used independently
- **W&B Integration**: Complete experiment tracking and model versioning
- **Multiple Models**: Compare different algorithms easily
- **Synthetic Data**: Realistic fraud patterns without sensitive real data
- **Configuration Driven**: Easy to modify experiments via YAML files
- **Class Imbalance Handling**: SMOTE resampling for fraud detection scenarios

## Dependencies

Core ML libraries: scikit-learn, XGBoost, LightGBM, imbalanced-learn
Experiment tracking: Weights & Biases
Data processing: pandas, numpy
Visualization: matplotlib, seaborn
Configuration: PyYAML