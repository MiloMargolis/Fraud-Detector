#!/usr/bin/env python3
"""
Main training script for fraud detection models.
Integrates data generation, model training, and W&B tracking.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_generator import FraudDataGenerator
from models.model_factory import ModelFactory
from utils.wandb_utils import (
    init_experiment, log_metrics, log_confusion_matrix, 
    log_classification_report, save_model_artifact, 
    log_feature_importance, log_data_info, finish_experiment
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class FraudDetectionTrainer:
    """
    Main trainer class for fraud detection models.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.data_generator = FraudDataGenerator(random_state=self.config['data']['random_state'])
        self.scaler = StandardScaler()
        self.resampler = None
        
    def load_config(self, config_path):
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'project': {'name': 'fraud-detection'},
                'data': {
                    'train_size': 0.8,
                    'test_size': 0.2,
                    'random_state': 42,
                    'fraud_ratio': 0.01,
                    'n_samples': 10000,
                    'use_resampling': True
                },
                'models': {
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    }
                },
                'training': {
                    'cv_folds': 5,
                    'scoring': ['precision', 'recall', 'f1', 'roc_auc'],
                    'save_model': True
                },
                'wandb': {
                    'log_artifacts': True,
                    'log_model': True,
                    'log_confusion_matrix': True
                }
            }
    
    def prepare_data(self):
        """
        Prepare the dataset for training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Generating synthetic fraud detection dataset...")
        
        # Generate dataset
        data, labels = self.data_generator.generate_dataset(
            n_samples=self.config['data']['n_samples'],
            fraud_ratio=self.config['data']['fraud_ratio']
        )
        
        # Split features and target
        X = data.drop('fraud', axis=1) if 'fraud' in data.columns else data
        y = labels if 'fraud' not in data.columns else data['fraud']
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Resampling if enabled
        if self.config['data'].get('use_resampling', True):
            print("Applying SMOTE resampling to balance classes...")
            smote = SMOTE(random_state=self.config['data']['random_state'])
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            print(f"Original training set: {len(X_train_scaled)} samples")
            print(f"Resampled training set: {len(X_train_resampled)} samples")
            print(f"Fraud ratio after resampling: {np.mean(y_train_resampled):.3f}")
            
            return X_train_resampled, X_test_scaled, y_train_resampled, y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_type, X_train, y_train, X_test, y_test):
        """
        Train a specific model and evaluate it.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Trained model and evaluation metrics
        """
        print(f"\nTraining {model_type} model...")
        
        # Get model parameters from config
        model_params = self.config['models'].get(model_type, {})
        
        # Create model
        model = ModelFactory.create_model(model_type, model_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model.model, X_train, y_train,
            cv=self.config['training']['cv_folds'],
            scoring='f1'
        )
        
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        print(f"Model performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return model, metrics, y_pred, y_pred_proba
    
    def log_results(self, model, metrics, y_test, y_pred, y_pred_proba, model_type):
        """
        Log results to W&B.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            y_test: True test labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            model_type: Type of model
        """
        # Log metrics
        log_metrics(metrics)
        
        # Log confusion matrix
        if self.config['wandb'].get('log_confusion_matrix', True):
            log_confusion_matrix(y_test, y_pred)
        
        # Log classification report
        log_classification_report(y_test, y_pred)
        
        # Log feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance:
            log_feature_importance(model.model, list(feature_importance.keys()), model_type)
        
        # Save model artifact
        if self.config['wandb'].get('log_model', True):
            save_model_artifact(model.model, f"{model_type}_fraud_detector")
    
    def run_experiment(self, model_type=None, run_name=None):
        """
        Run a complete experiment.
        
        Args:
            model_type: Type of model to train (if None, use config)
            run_name: Name for the W&B run
        """
        # Initialize W&B experiment
        run = init_experiment(
            config_path=None,  # Config already loaded
            project_name=self.config['project']['name'],
            run_name=run_name or f"{model_type}_experiment"
        )
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            # Log data information
            log_data_info(X_train, X_test, y_train, y_test)
            
            # Determine model type
            if model_type is None:
                model_type = list(self.config['models'].keys())[0]
            
            # Train model
            model, metrics, y_pred, y_pred_proba = self.train_model(
                model_type, X_train, y_train, X_test, y_test
            )
            
            # Log results
            self.log_results(model, metrics, y_test, y_pred, y_pred_proba, model_type)
            
            print(f"\nExperiment completed successfully!")
            print(f"Best F1 Score: {metrics['f1_score']:.4f}")
            print(f"Best ROC AUC: {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            raise
        finally:
            finish_experiment()
    
    def run_multiple_models(self):
        """
        Run experiments for multiple models.
        """
        print("Running experiments for multiple models...")
        
        # Prepare data once
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        results = {}
        
        for model_type in self.config['models'].keys():
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*50}")
            
            # Initialize W&B experiment for this model
            run = init_experiment(
                project_name=self.config['project']['name'],
                run_name=f"{model_type}_experiment"
            )
            
            try:
                # Train model
                model, metrics, y_pred, y_pred_proba = self.train_model(
                    model_type, X_train, y_train, X_test, y_test
                )
                
                # Log results
                self.log_results(model, metrics, y_test, y_pred, y_pred_proba, model_type)
                
                results[model_type] = metrics
                
                print(f"{model_type} completed successfully!")
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
            finally:
                finish_experiment()
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        for model_type, metrics in results.items():
            print(f"\n{model_type.upper()}:")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")


def main():
    """
    Main function to run training experiments.
    """
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Specific model to train')
    parser.add_argument('--run-name', type=str, help='Name for the W&B run')
    parser.add_argument('--multi', action='store_true', help='Train multiple models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FraudDetectionTrainer(config_path=args.config)
    
    if args.multi:
        # Train multiple models
        trainer.run_multiple_models()
    else:
        # Train single model
        trainer.run_experiment(model_type=args.model, run_name=args.run_name)


if __name__ == "__main__":
    main() 