"""
Model factory for fraud detection models.
Provides different ML algorithms with consistent interfaces.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin


class FraudDetectionModel:
    """
    Base class for fraud detection models with consistent interface.
    """
    
    def __init__(self, model_name, model_params=None):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model
            model_params: Dictionary of model parameters
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Args:
            X: Training features
            y: Training labels
        """
        raise NotImplementedError
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            return None


class RandomForestModel(FraudDetectionModel):
    """
    Random Forest model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("random_forest", default_params)
        self.model = RandomForestClassifier(**self.model_params)
    
    def fit(self, X, y):
        """Fit the Random Forest model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class XGBoostModel(FraudDetectionModel):
    """
    XGBoost model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': 99  # For imbalanced dataset (1% fraud)
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("xgboost", default_params)
        self.model = xgb.XGBClassifier(**self.model_params)
    
    def fit(self, X, y):
        """Fit the XGBoost model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class LightGBMModel(FraudDetectionModel):
    """
    LightGBM model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("lightgbm", default_params)
        self.model = lgb.LGBMClassifier(**self.model_params)
    
    def fit(self, X, y):
        """Fit the LightGBM model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class LogisticRegressionModel(FraudDetectionModel):
    """
    Logistic Regression model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("logistic_regression", default_params)
        self.model = LogisticRegression(**self.model_params)
    
    def fit(self, X, y):
        """Fit the Logistic Regression model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class SVMModel(FraudDetectionModel):
    """
    Support Vector Machine model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42,
            'class_weight': 'balanced',
            'probability': True
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("svm", default_params)
        self.model = SVC(**self.model_params)
    
    def fit(self, X, y):
        """Fit the SVM model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class NeuralNetworkModel(FraudDetectionModel):
    """
    Neural Network model for fraud detection.
    """
    
    def __init__(self, model_params=None):
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__("neural_network", default_params)
        self.model = MLPClassifier(**self.model_params)
    
    def fit(self, X, y):
        """Fit the Neural Network model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class ModelFactory:
    """
    Factory class for creating fraud detection models.
    """
    
    @staticmethod
    def create_model(model_type, model_params=None):
        """
        Create a model instance based on the model type.
        
        Args:
            model_type: Type of model to create
            model_params: Model parameters
            
        Returns:
            Model instance
        """
        model_classes = {
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'logistic_regression': LogisticRegressionModel,
            'svm': SVMModel,
            'neural_network': NeuralNetworkModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(model_classes.keys())}")
        
        return model_classes[model_type](model_params)
    
    @staticmethod
    def get_available_models():
        """
        Get list of available model types.
        
        Returns:
            List of available model types
        """
        return ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'svm', 'neural_network']
    
    @staticmethod
    def get_default_params(model_type):
        """
        Get default parameters for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        model = ModelFactory.create_model(model_type)
        return model.model_params


def main():
    """
    Example usage of the model factory.
    """
    # Create different models
    models = {}
    
    for model_type in ModelFactory.get_available_models():
        print(f"Creating {model_type} model...")
        models[model_type] = ModelFactory.create_model(model_type)
    
    print(f"\nCreated {len(models)} models:")
    for name, model in models.items():
        print(f"- {name}: {type(model.model).__name__}")


if __name__ == "__main__":
    main() 