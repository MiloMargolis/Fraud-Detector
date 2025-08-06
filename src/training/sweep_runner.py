#!/usr/bin/env python3
"""
W&B Sweep runner for hyperparameter optimization.
Uses native W&B sweep functionality for automated hyperparameter tuning.
"""

import wandb
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_generator import FraudDataGenerator
from models.model_factory import ModelFactory
from utils.wandb_utils import (
    log_metrics, log_confusion_matrix, log_classification_report,
    save_model_artifact, log_feature_importance, log_data_info,
    log_roc_curve, log_precision_recall_curve, log_data_tables
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE


def train_sweep_model():
    """
    Training function for W&B sweep.
    This function will be called by the sweep agent.
    """
    # Initialize wandb run for sweep
    wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    try:
        # Generate data
        generator = FraudDataGenerator(random_state=42)
        data, labels = generator.generate_dataset(
            n_samples=10000,
            fraud_ratio=0.01,
            add_noise=True
        )
        
        # Split features and target
        X = data.drop('fraud', axis=1) if 'fraud' in data.columns else data
        y = labels if 'fraud' not in data.columns else data['fraud']
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Apply SMOTE resampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Log data information
        log_data_info(X_train_resampled, X_test_scaled, y_train_resampled, y_test)
        
        # Create model with sweep parameters
        model_params = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'min_samples_split': config.min_samples_split,
            'min_samples_leaf': config.min_samples_leaf,
            'random_state': 42
        }
        
        # Create and train model
        model = ModelFactory.create_model('random_forest', model_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model.model, X_train_resampled, y_train_resampled,
            cv=5,
            scoring='f1'
        )
        
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        # Log all results using native W&B features
        log_metrics(metrics)
        log_confusion_matrix(y_test, y_pred)
        log_classification_report(y_test, y_pred)
        log_roc_curve(y_test, y_pred_proba)
        log_precision_recall_curve(y_test, y_pred_proba)
        
        # Log feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance:
            log_feature_importance(model.model, list(feature_importance.keys()), 'random_forest')
        
        # Log data tables
        log_data_tables(X_train_resampled, X_test_scaled, y_train_resampled, y_test)
        
        # Save model artifact
        save_model_artifact(model.model, f"random_forest_sweep_{wandb.run.id}")
        
        print(f"Sweep run completed with F1 Score: {metrics['f1_score']:.4f}")
        
    except Exception as e:
        print(f"Error in sweep run: {e}")
        wandb.log({"error": str(e)})
    
    finally:
        wandb.finish()


def create_sweep_config():
    """
    Create W&B sweep configuration for hyperparameter optimization.
    
    Returns:
        Dictionary with sweep configuration
    """
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {
            "name": "f1_score",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "min": 50,
                "max": 300,
                "distribution": "int_uniform"
            },
            "max_depth": {
                "min": 3,
                "max": 15,
                "distribution": "int_uniform"
            },
            "min_samples_split": {
                "min": 2,
                "max": 20,
                "distribution": "int_uniform"
            },
            "min_samples_leaf": {
                "min": 1,
                "max": 10,
                "distribution": "int_uniform"
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        }
    }
    
    return sweep_config


def run_sweep():
    """
    Run the W&B sweep for hyperparameter optimization.
    """
    # Initialize wandb
    wandb.init(project="fraud-detection", name="hyperparameter_sweep")
    
    # Create sweep configuration
    sweep_config = create_sweep_config()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="fraud-detection")
    
    print(f"Sweep created with ID: {sweep_id}")
    print("Starting sweep agent...")
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=train_sweep_model, count=20)
    
    print("Sweep completed!")
    print(f"View results at: https://wandb.ai/{wandb.run.entity}/fraud-detection/sweeps/{sweep_id}")


def main():
    """
    Main function to run hyperparameter sweep.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run W&B sweep for hyperparameter optimization')
    parser.add_argument('--sweep-id', type=str, help='Existing sweep ID to continue')
    
    args = parser.parse_args()
    
    if args.sweep_id:
        # Continue existing sweep
        print(f"Continuing sweep: {args.sweep_id}")
        wandb.agent(args.sweep_id, function=train_sweep_model, count=10)
    else:
        # Create and run new sweep
        run_sweep()


if __name__ == "__main__":
    main() 