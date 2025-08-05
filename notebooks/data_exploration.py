#!/usr/bin/env python3
"""
Data exploration script for fraud detection dataset.
Provides insights into data distribution and patterns.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_generator import FraudDataGenerator

def main():
    """Main exploration function."""
    print("Fraud Detection Data Exploration")
    print("=" * 40)
    
    # Generate sample dataset
    generator = FraudDataGenerator(random_state=42)
    data, labels = generator.generate_dataset(
        n_samples=10000,
        fraud_ratio=0.01,
        add_noise=True
    )
    
    # Add labels to dataframe
    data['fraud'] = labels
    
    print(f"Dataset shape: {data.shape}")
    print(f"Fraud ratio: {np.mean(labels):.3f}")
    print(f"Number of features: {data.shape[1] - 1}")
    
    # Basic statistics
    print("\nDataset Info:")
    print(data.info())
    
    print("\nStatistical Summary:")
    print(data.describe())
    
    # Target distribution
    fraud_counts = data['fraud'].value_counts()
    print(f"\nTarget Distribution:")
    print(f"Legitimate transactions: {fraud_counts[0]}")
    print(f"Fraudulent transactions: {fraud_counts[1]}")
    print(f"Fraud percentage: {np.mean(labels)*100:.2f}%")
    
    # Feature correlations with fraud
    numerical_features = ['amount', 'customer_age', 'account_age', 'prev_transactions', 
                         'avg_amount', 'time_since_last', 'amount_deviation', 
                         'transaction_frequency', 'account_age_years']
    
    correlations = data[numerical_features + ['fraud']].corr()['fraud'].sort_values(ascending=False)
    
    print(f"\nTop correlations with fraud:")
    for feature, corr in correlations.head(10).items():
        if feature != 'fraud':
            print(f"  {feature}: {corr:.4f}")
    
    # Amount analysis
    print(f"\nAmount Analysis:")
    print(f"Average legitimate transaction: ${data[data['fraud']==0]['amount'].mean():.2f}")
    print(f"Average fraudulent transaction: ${data[data['fraud']==1]['amount'].mean():.2f}")
    print(f"Amount difference: ${data[data['fraud']==1]['amount'].mean() - data[data['fraud']==0]['amount'].mean():.2f}")
    
    # Time analysis
    print(f"\nTime Analysis:")
    late_night_fraud = data[(data['is_late_night']==1) & (data['fraud']==1)].shape[0]
    late_night_total = data[data['is_late_night']==1].shape[0]
    print(f"Late night fraud rate: {late_night_fraud/late_night_total:.3f}")
    
    new_customer_fraud = data[(data['is_new_customer']==1) & (data['fraud']==1)].shape[0]
    new_customer_total = data[data['is_new_customer']==1].shape[0]
    print(f"New customer fraud rate: {new_customer_fraud/new_customer_total:.3f}")
    
    print("\nExploration completed!")

if __name__ == "__main__":
    main() 