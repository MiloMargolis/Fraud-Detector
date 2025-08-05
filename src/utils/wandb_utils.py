"""
Weights & Biases utilities for the Fraud Detection project.
Helper functions for experiment tracking, logging, and artifact management.
"""

import wandb
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def init_experiment(config_path=None, project_name="fraud-detection", run_name=None):
    """
    Initialize a W&B experiment with configuration.
    
    Args:
        config_path: Path to YAML configuration file
        project_name: Name of the W&B project
        run_name: Name for this specific run
    
    Returns:
        wandb run object
    """
    config = {}
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=["fraud-detection", "classification"]
    )
    
    return run


def log_model_parameters(model, model_name):
    """
    Log model parameters to W&B.
    
    Args:
        model: Trained model object
        model_name: Name of the model
    """
    if hasattr(model, 'get_params'):
        params = model.get_params()
        wandb.log({f"{model_name}_params": wandb.Table(data=[[k, str(v)] for k, v in params.items()],
                                                      columns=["Parameter", "Value"])})


def log_metrics(metrics_dict, step=None):
    """
    Log metrics to W&B.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Step number for the metrics
    """
    wandb.log(metrics_dict, step=step)


def log_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Log confusion matrix to W&B.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    if class_names is None:
        class_names = ["Legitimate", "Fraud"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Log to W&B
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


def log_classification_report(y_true, y_pred, target_names=None):
    """
    Log classification report to W&B.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the target classes
    """
    if target_names is None:
        target_names = ["Legitimate", "Fraud"]
    
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Log individual metrics
    for class_name in target_names:
        if class_name in report:
            wandb.log({
                f"{class_name.lower()}_precision": report[class_name]['precision'],
                f"{class_name.lower()}_recall": report[class_name]['recall'],
                f"{class_name.lower()}_f1_score": report[class_name]['f1-score']
            })
    
    # Log overall metrics
    wandb.log({
        "accuracy": report['accuracy'],
        "macro_avg_precision": report['macro avg']['precision'],
        "macro_avg_recall": report['macro avg']['recall'],
        "macro_avg_f1": report['macro avg']['f1-score']
    })


def save_model_artifact(model, model_name, model_path="models"):
    """
    Save model as W&B artifact.
    
    Args:
        model: Trained model object
        model_name: Name for the model
        model_path: Directory to save the model
    """
    Path(model_path).mkdir(exist_ok=True)
    
    # Save model locally
    model_file = f"{model_path}/{model_name}.joblib"
    joblib.dump(model, model_file)
    
    # Create W&B artifact
    artifact = wandb.Artifact(
        name=f"{model_name}-model",
        type="model",
        description=f"Trained {model_name} model for fraud detection"
    )
    
    artifact.add_file(model_file)
    wandb.log_artifact(artifact)
    
    return model_file


def log_feature_importance(model, feature_names, model_name):
    """
    Log feature importance to W&B.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log as table
        wandb.log({f"{model_name}_feature_importance": wandb.Table(dataframe=importance_df)})
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} Feature Importance')
        plt.gca().invert_yaxis()
        
        wandb.log({f"{model_name}_feature_importance_plot": wandb.Image(plt)})
        plt.close()


def log_data_info(X_train, X_test, y_train, y_test):
    """
    Log dataset information to W&B.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    data_info = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
        "train_fraud_ratio": np.mean(y_train),
        "test_fraud_ratio": np.mean(y_test),
        "train_legitimate": np.sum(y_train == 0),
        "train_fraud": np.sum(y_train == 1),
        "test_legitimate": np.sum(y_test == 0),
        "test_fraud": np.sum(y_test == 1)
    }
    
    wandb.log(data_info)


def create_sweep_config():
    """
    Create W&B sweep configuration for hyperparameter optimization.
    
    Returns:
        Dictionary with sweep configuration
    """
    sweep_config = {
        "method": "bayes",
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
        }
    }
    
    return sweep_config


def finish_experiment():
    """
    Finish the current W&B experiment.
    """
    wandb.finish() 