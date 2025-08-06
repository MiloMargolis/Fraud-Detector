# Setup and Testing Guide

This guide will help you set up and test the fraud detection system step by step.

## 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Test Data Generation (No W&B Required)

Test the synthetic data generation to see fraud patterns:

```bash
python notebooks/data_exploration.py
```

This will:
- Generate 10,000 synthetic transactions with 1% fraud ratio
- Show dataset statistics and feature correlations
- Display fraud patterns and data distribution

## 3. Test Model Factory (No W&B Required)

Verify all models can be created successfully:

```bash
python src/models/model_factory.py
```

This will:
- Create instances of all available models (Random Forest, XGBoost, LightGBM, etc.)
- Show that the model factory is working correctly

## 4. Set Up Weights & Biases (Optional but Recommended)

For full experiment tracking:

```bash
wandb login
python src/init_wandb.py
```

This will:
- Create W&B project configuration
- Generate default experiment configs
- Set up project structure

## 5. Run Single Model Training with Native W&B Features

Train one model with enhanced W&B tracking:

```bash
python src/training/train.py --model random_forest
```

This will:
- Generate synthetic data
- Train a Random Forest model
- Log metrics using native W&B features:
  - Interactive confusion matrix
  - ROC curve and precision-recall curve
  - Feature importance bar charts
  - Interactive data tables
  - Classification report tables
- Save the trained model as W&B artifact

## 6. Run Multiple Models Comparison

Compare all available models:

```bash
python src/training/train.py --multi
```

This will:
- Train Random Forest, XGBoost, LightGBM, and other models
- Compare performance across all models
- Log all results using native W&B visualizations
- Show summary of best performing models

## 7. Run W&B Sweeps (Hyperparameter Optimization)

Automated hyperparameter tuning using W&B Sweeps:

```bash
python src/training/sweep_runner.py
```

This will:
- Create a Bayesian optimization sweep
- Automatically test different hyperparameter combinations
- Track all experiments in W&B
- Find the best hyperparameters for your model
- Show sweep results and comparisons

## 8. View Results

After training, view your experiments:
- Go to https://wandb.ai
- Find your "fraud-detection" project
- Explore:
  - Interactive charts and visualizations
  - Model comparison tables
  - Sweep results and parameter importance
  - Confusion matrices and ROC curves
  - Feature importance rankings
  - Data samples in interactive tables

## Testing Order

**Start with steps 1-3** to verify everything works without W&B, then proceed with W&B integration for the full experience.

## New W&B Features Demonstrated

- **Native Confusion Matrix**: Interactive confusion matrix visualization
- **ROC and PR Curves**: Automatic curve plotting with W&B
- **Interactive Tables**: Data exploration with sortable tables
- **Feature Importance**: Bar charts showing feature rankings
- **W&B Sweeps**: Automated hyperparameter optimization
- **Model Artifacts**: Versioned model storage and tracking
- **Classification Reports**: Interactive performance tables

## Troubleshooting

- If you get import errors, make sure your virtual environment is activated
- If W&B login fails, create an account at https://wandb.ai first
- If models fail to train, check that all dependencies are installed correctly
- For XGBoost errors on macOS, run: `brew install libomp` 