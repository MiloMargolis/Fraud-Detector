#!/usr/bin/env python3
"""
Weights & Biases initialization script for the Fraud Detection project.
This script sets up the W&B project and provides basic configuration.
"""

import wandb
import yaml
import os
from pathlib import Path

def create_wandb_project():
    """Initialize the W&B project for fraud detection."""
    
    # Project configuration
    project_config = {
        "project_name": "fraud-detection",
        "entity": None,  # Will use your default entity
        "description": "Fraud detection system using various ML algorithms",
        "tags": ["fraud-detection", "machine-learning", "classification"]
    }
    
    # Initialize W&B project
    try:
        wandb.init(
            project=project_config["project_name"],
            entity=project_config["entity"],
            config={
                "project_description": project_config["description"],
                "tags": project_config["tags"]
            }
        )
        
        print(f" Successfully initialized W&B project: {project_config['project_name']}")
        print(f"🔗 View your project at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
        
        # Log project metadata
        wandb.run.summary["project_type"] = "fraud-detection"
        wandb.run.summary["framework"] = "scikit-learn"
        
        wandb.finish()
        
    except Exception as e:
        print(f" Error initializing W&B project: {e}")
        print("Make sure you're logged in with 'wandb login'")

def create_default_configs():
    """Create default configuration files for experiments."""
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Default experiment configuration
    default_config = {
        "project": {
            "name": "fraud-detection",
            "entity": None
        },
        "data": {
            "train_size": 0.8,
            "test_size": 0.2,
            "random_state": 42,
            "fraud_ratio": 0.01  # 1% fraud cases (realistic for fraud detection)
        },
        "models": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
        },
        "training": {
            "cv_folds": 5,
            "scoring": ["precision", "recall", "f1", "roc_auc"],
            "save_model": True
        },
        "wandb": {
            "log_artifacts": True,
            "log_model": True,
            "log_confusion_matrix": True
        }
    }
    
    # Save default config
    with open(configs_dir / "default.yaml", "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    # Create experiment-specific configs
    experiments = {
        "experiment_1": {
            "description": "Baseline Random Forest",
            "model": "random_forest",
            "data": {"fraud_ratio": 0.01}
        },
        "experiment_2": {
            "description": "XGBoost with default params",
            "model": "xgboost",
            "data": {"fraud_ratio": 0.01}
        },
        "experiment_3": {
            "description": "LightGBM with default params",
            "model": "lightgbm",
            "data": {"fraud_ratio": 0.01}
        }
    }
    
    for exp_name, exp_config in experiments.items():
        config = default_config.copy()
        config.update(exp_config)
        
        with open(configs_dir / f"{exp_name}.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created configuration files in {configs_dir}")

def create_gitignore():
    """Create .gitignore file for the project."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/
data/processed/
*.csv
*.json
*.parquet

# Model files
models/*.pkl
models/*.joblib
models/*.h5

# W&B
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment variables
.env
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("Created .gitignore file")

def main():
    """Main function to initialize the project."""
    
    print("🚀 Initializing Fraud Detection Project with Weights & Biases")
    print("=" * 60)
    
    # Create project structure
    create_default_configs()
    create_gitignore()
    
    # Initialize W&B project
    print("\n📊 Setting up Weights & Biases project...")
    create_wandb_project()
    
    print("\n Project initialization complete!")
    print("\n Next steps:")
    print("1. Run 'wandb login' to authenticate")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start with: python src/training/train.py")
    print("4. View experiments at: https://wandb.ai")

if __name__ == "__main__":
    main() 