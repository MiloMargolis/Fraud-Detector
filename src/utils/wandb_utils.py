"""
Weights & Biases utilities for the Fraud Detection project.
Uses native W&B features.
"""

import wandb
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


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


def log_metrics(metrics_dict, step=None):
    """
    Log metrics to W&B using native logging.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Step number for the metrics
    """
    wandb.log(metrics_dict, step=step)


def log_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Log confusion matrix using W&B's native confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    if class_names is None:
        class_names = ["Legitimate", "Fraud"]
    
    # Use W&B's native confusion matrix
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )})


def log_classification_report(y_true, y_pred, target_names=None):
    """
    Log classification report using W&B's native table.
    
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
    
    # Log classification report as table
    report_data = []
    for class_name in target_names:
        if class_name in report:
            report_data.append([
                class_name,
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score'],
                report[class_name]['support']
            ])
    
    # Add overall metrics
    report_data.append([
        "Overall",
        report['accuracy'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score'],
        len(y_true)
    ])
    
    wandb.log({"classification_report": wandb.Table(
        data=report_data,
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
    )})


def log_roc_curve(y_true, y_pred_proba):
    """
    Log ROC curve using W&B's native plotting.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
    """
    # W&B expects probabilities for each class, so we need to reshape
    # For binary classification, we need [prob_class_0, prob_class_1]
    y_pred_proba_reshaped = np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    wandb.log({"roc_curve": wandb.plot.roc_curve(
        y_true,
        y_pred_proba_reshaped,
        labels=["Legitimate", "Fraud"]
    )})


def log_precision_recall_curve(y_true, y_pred_proba):
    """
    Log precision-recall curve using W&B's native plotting.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
    """
    # W&B expects probabilities for each class, so we need to reshape
    # For binary classification, we need [prob_class_0, prob_class_1]
    y_pred_proba_reshaped = np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    wandb.log({"precision_recall_curve": wandb.plot.pr_curve(
        y_true,
        y_pred_proba_reshaped,
        labels=["Legitimate", "Fraud"]
    )})


def save_model_artifact(model, model_name, model_path="models"):
    """
    Save model as W&B artifact using native artifact system.
    
    Args:
        model: Trained model object
        model_name: Name for the model
        model_path: Directory to save the model
    """
    Path(model_path).mkdir(exist_ok=True)
    
    # Save model locally
    model_file = f"{model_path}/{model_name}.joblib"
    joblib.dump(model, model_file)
    
    # Create W&B artifact using native system
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
    Log feature importance using W&B's native bar chart.
    
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
        
        # Log as W&B table
        wandb.log({f"{model_name}_feature_importance": wandb.Table(
            dataframe=importance_df.head(20)
        )})
        
        # Log as W&B bar chart
        wandb.log({f"{model_name}_feature_importance_chart": wandb.plot.bar(
            wandb.Table(data=importance_df.head(20).values.tolist(), 
                       columns=["Feature", "Importance"]),
            "Feature",
            "Importance",
            title=f"{model_name} Feature Importance"
        )})


def log_data_info(X_train, X_test, y_train, y_test):
    """
    Log dataset information using W&B's native logging.
    
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


def log_data_tables(X_train, X_test, y_train, y_test, sample_size=1000):
    """
    Log data samples as W&B tables for interactive exploration.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        sample_size: Number of samples to log
    """
    # Sample data for visualization - use integer indices to avoid index issues
    train_indices = np.random.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
    test_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    
    train_sample = X_train.iloc[train_indices].copy()
    test_sample = X_test.iloc[test_indices].copy()
    
    # Add labels - handle both pandas and numpy arrays
    if hasattr(y_train, 'iloc'):
        train_sample['fraud'] = y_train.iloc[train_indices]
    else:
        train_sample['fraud'] = y_train[train_indices]
        
    if hasattr(y_test, 'iloc'):
        test_sample['fraud'] = y_test.iloc[test_indices]
    else:
        test_sample['fraud'] = y_test[test_indices]
    
    # Log as W&B tables
    wandb.log({"training_data_sample": wandb.Table(dataframe=train_sample)})
    wandb.log({"test_data_sample": wandb.Table(dataframe=test_sample)})


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