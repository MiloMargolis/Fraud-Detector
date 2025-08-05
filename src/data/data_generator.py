"""
Data generator for synthetic fraud detection dataset.
Creates realistic transaction data with fraud patterns.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import random


class FraudDataGenerator:
    """
    Generator for synthetic fraud detection data.
    Creates realistic transaction data with various fraud patterns.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_transaction_features(self, n_samples):
        """
        Generate realistic transaction features.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with transaction features
        """
        # Transaction amount (log-normal distribution for realistic amounts)
        amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n_samples)
        amounts = np.clip(amounts, 1, 10000)  # Clip to reasonable range
        
        # Time features (hour of day, day of week)
        hours = np.random.randint(0, 24, size=n_samples)
        days = np.random.randint(0, 7, size=n_samples)
        
        # Merchant categories (encoded as integers)
        merchant_categories = np.random.randint(0, 20, size=n_samples)
        
        # Location features (latitude, longitude)
        lat = np.random.uniform(25, 50, size=n_samples)  # US latitude range
        lon = np.random.uniform(-125, -65, size=n_samples)  # US longitude range
        
        # Customer features
        customer_age = np.random.normal(45, 15, size=n_samples)
        customer_age = np.clip(customer_age, 18, 80)
        
        # Account age (days since account creation)
        account_age = np.random.exponential(365, size=n_samples)
        account_age = np.clip(account_age, 1, 3650)  # 1 day to 10 years
        
        # Previous transaction count
        prev_transactions = np.random.poisson(50, size=n_samples)
        
        # Average transaction amount for this customer
        avg_amount = np.random.lognormal(mean=4.0, sigma=0.8, size=n_samples)
        
        # Time since last transaction (hours)
        time_since_last = np.random.exponential(24, size=n_samples)
        time_since_last = np.clip(time_since_last, 0, 720)  # 0 to 30 days
        
        # Create DataFrame
        data = pd.DataFrame({
            'amount': amounts,
            'hour': hours,
            'day_of_week': days,
            'merchant_category': merchant_categories,
            'latitude': lat,
            'longitude': lon,
            'customer_age': customer_age,
            'account_age': account_age,
            'prev_transactions': prev_transactions,
            'avg_amount': avg_amount,
            'time_since_last': time_since_last
        })
        
        return data
    
    def generate_fraud_patterns(self, data, fraud_ratio=0.01):
        """
        Generate fraud labels based on realistic patterns.
        
        Args:
            data: Transaction features DataFrame
            fraud_ratio: Ratio of fraudulent transactions
            
        Returns:
            Array of fraud labels (0: legitimate, 1: fraud)
        """
        n_samples = len(data)
        n_fraud = int(n_samples * fraud_ratio)
        
        # Initialize all as legitimate
        labels = np.zeros(n_samples)
        
        # Fraud patterns based on realistic scenarios
        fraud_indices = []
        
        # Pattern 1: Unusually large amounts
        large_amount_threshold = np.percentile(data['amount'], 95)
        large_amount_indices = data[data['amount'] > large_amount_threshold].index
        fraud_indices.extend(np.random.choice(large_amount_indices, 
                                            size=min(len(large_amount_indices), n_fraud // 4), 
                                            replace=False))
        
        # Pattern 2: Unusual time patterns (late night transactions)
        late_night_indices = data[data['hour'].isin([0, 1, 2, 3, 4, 5])].index
        fraud_indices.extend(np.random.choice(late_night_indices, 
                                            size=min(len(late_night_indices), n_fraud // 4), 
                                            replace=False))
        
        # Pattern 3: New accounts with large transactions
        new_account_threshold = 30  # 30 days
        new_account_indices = data[data['account_age'] < new_account_threshold].index
        fraud_indices.extend(np.random.choice(new_account_indices, 
                                            size=min(len(new_account_indices), n_fraud // 4), 
                                            replace=False))
        
        # Pattern 4: Unusual location patterns (far from customer's usual location)
        # Simulate by selecting random transactions
        remaining_indices = list(set(range(n_samples)) - set(fraud_indices))
        remaining_fraud_needed = n_fraud - len(fraud_indices)
        
        if remaining_fraud_needed > 0 and remaining_indices:
            additional_fraud_indices = np.random.choice(remaining_indices, 
                                                      size=min(remaining_fraud_needed, len(remaining_indices)), 
                                                      replace=False)
            fraud_indices.extend(additional_fraud_indices)
        
        # Ensure we have exactly n_fraud fraudulent transactions
        fraud_indices = list(set(fraud_indices))[:n_fraud]
        
        # Mark fraudulent transactions
        labels[fraud_indices] = 1
        
        return labels
    
    def add_noise_features(self, data, n_noise_features=5):
        """
        Add noise features to make the dataset more realistic.
        
        Args:
            data: Original features DataFrame
            n_noise_features: Number of noise features to add
            
        Returns:
            DataFrame with added noise features
        """
        for i in range(n_noise_features):
            # Random noise features
            noise = np.random.normal(0, 1, size=len(data))
            data[f'noise_feature_{i}'] = noise
            
        return data
    
    def generate_dataset(self, n_samples=10000, fraud_ratio=0.01, add_noise=True):
        """
        Generate complete fraud detection dataset.
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent transactions
            add_noise: Whether to add noise features
            
        Returns:
            Tuple of (features, labels)
        """
        print(f"Generating {n_samples} transactions with {fraud_ratio*100:.1f}% fraud ratio...")
        
        # Generate transaction features
        data = self.generate_transaction_features(n_samples)
        
        # Generate fraud labels
        labels = self.generate_fraud_patterns(data, fraud_ratio)
        
        # Add noise features if requested
        if add_noise:
            data = self.add_noise_features(data)
        
        # Add some engineered features
        data = self.add_engineered_features(data)
        
        print(f"Generated dataset with {np.sum(labels)} fraudulent transactions")
        print(f"Features shape: {data.shape}")
        
        return data, labels
    
    def add_engineered_features(self, data):
        """
        Add engineered features that are useful for fraud detection.
        
        Args:
            data: Original features DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Amount deviation from customer average
        data['amount_deviation'] = (data['amount'] - data['avg_amount']) / (data['avg_amount'] + 1e-8)
        
        # Transaction frequency (inverse of time since last)
        data['transaction_frequency'] = 1 / (data['time_since_last'] + 1e-8)
        
        # Account age in years
        data['account_age_years'] = data['account_age'] / 365
        
        # Time-based features
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_late_night'] = (data['hour'] <= 6).astype(int)
        data['is_business_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype(int)
        
        # Amount-based features
        data['is_large_amount'] = (data['amount'] > np.percentile(data['amount'], 90)).astype(int)
        data['is_small_amount'] = (data['amount'] < np.percentile(data['amount'], 10)).astype(int)
        
        # Customer behavior features
        data['is_new_customer'] = (data['account_age'] < 30).astype(int)
        data['is_active_customer'] = (data['prev_transactions'] > np.median(data['prev_transactions'])).astype(int)
        
        return data
    
    def save_dataset(self, data, labels, filepath):
        """
        Save the generated dataset to a file.
        
        Args:
            data: Features DataFrame
            labels: Target labels
            filepath: Path to save the dataset
        """
        # Combine features and labels
        dataset = data.copy()
        dataset['fraud'] = labels
        
        # Save to CSV
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        
        return dataset


def main():
    """
    Main function to generate and save a sample dataset.
    """
    # Initialize generator
    generator = FraudDataGenerator(random_state=42)
    
    # Generate dataset
    data, labels = generator.generate_dataset(
        n_samples=10000,
        fraud_ratio=0.01,
        add_noise=True
    )
    
    # Save dataset
    Path("data").mkdir(exist_ok=True)
    generator.save_dataset(data, labels, "data/fraud_dataset.csv")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Fraudulent transactions: {np.sum(labels)}")
    print(f"Legitimate transactions: {np.sum(labels == 0)}")
    print(f"Fraud ratio: {np.mean(labels):.3f}")
    print(f"Number of features: {data.shape[1]}")


if __name__ == "__main__":
    from pathlib import Path
    main() 