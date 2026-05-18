"""
Data Generator for House Price Prediction
Generates synthetic house data with realistic features and prices
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

def generate_house_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic house price data
    
    Parameters:
    n_samples: Number of houses to generate
    random_seed: For reproducibility
    
    Returns:
    DataFrame with house features and prices
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Generate features
    area = np.random.normal(1500, 500, n_samples)  # Square feet
    area = np.clip(area, 500, 4000)  # Clip to realistic range
    
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples)
    bathrooms = np.round(bathrooms * 2) / 2  # Round to 0.5 increments
    
    age = np.random.exponential(20, n_samples)  # Age in years
    age = np.clip(age, 0, 50)
    
    # Location score (0-100)
    location_score = np.random.normal(70, 15, n_samples)
    location_score = np.clip(location_score, 0, 100)
    
    # Garage (0,1,2 cars)
    garage = np.random.choice([0, 1, 2], n_samples, p=[0.15, 0.65, 0.20])
    
    # Year built (1950-2023)
    current_year = datetime.now().year
    year_built = np.random.randint(1950, current_year + 1, n_samples)
    
    # Calculate price based on features (with some noise)
    base_price = 50000
    
    # Feature coefficients (realistic relationships)
    area_coef = 80  # $80 per sq ft
    bedroom_coef = 15000
    bathroom_coef = 12000
    age_coef = -800  # Older houses cheaper
    location_coef = 1500
    garage_coef = 8000
    
    # Calculate price
    price = (base_price +
             area * area_coef +
             bedrooms * bedroom_coef +
             bathrooms * bathroom_coef +
             age * age_coef +
             location_score * location_coef +
             garage * garage_coef)
    
    # Add random noise (10% variation)
    noise = np.random.normal(0, price * 0.1, n_samples)
    price = price + noise
    
    # Ensure no negative prices
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'area': np.round(area, 0).astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': np.round(age, 1),
        'location_score': np.round(location_score, 1),
        'garage': garage,
        'year_built': year_built,
        'price': np.round(price, 0).astype(int)
    })
    
    return df

def save_data(df, filename='house_data.csv'):
    """Save DataFrame to CSV"""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Shape: {df.shape}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    return filename

if __name__ == "__main__":
    # Generate 2000 samples
    print("Generating house price data...")
    data = generate_house_data(n_samples=2000)
    
    # Save to CSV
    save_data(data)
    
    # Display first few rows
    print("\nFirst 5 rows of generated data:")
    print(data.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(data.describe())