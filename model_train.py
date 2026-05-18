"""
Model Training for House Price Prediction
Trains a linear regression model and saves it for use in the web app
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def load_and_prepare_data(self, data_path='house_data.csv'):
        """Load and prepare data for training"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Select features for modeling
        self.feature_columns = ['area', 'bedrooms', 'bathrooms', 'age', 
                                'location_score', 'garage']
        
        X = df[self.feature_columns]
        y = df['price']
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {self.feature_columns}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, use_regularization=False):
        """Train the linear regression model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if use_regularization:
            print("Using Ridge Regression (with regularization)...")
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            print("Using Linear Regression...")
            self.model = LinearRegression()
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Evaluate model
        metrics = self.evaluate_model(y_train, y_train_pred, y_test, y_test_pred)
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        
        return metrics, feature_importance
    
    def evaluate_model(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate evaluation metrics"""
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_train_pred),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mse': mean_squared_error(y_train, y_train_pred)
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mse': mean_squared_error(y_test, y_test_pred)
            }
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print("\nTraining Set:")
        for metric, value in metrics['train'].items():
            print(f"  {metric.upper()}: {value:.2f}")
        
        print("\nTest Set:")
        for metric, value in metrics['test'].items():
            print(f"  {metric.upper()}: {value:.2f}")
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance (coefficients)"""
        if self.model is None:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_
        })
        importance['abs_coefficient'] = np.abs(importance['coefficient'])
        importance = importance.sort_values('abs_coefficient', ascending=False)
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (Coefficients)")
        print("="*50)
        for _, row in importance.iterrows():
            print(f"  {row['feature']:15s}: {row['coefficient']:8.2f}")
        
        return importance
    
    def plot_predictions(self, X_test, y_test, save_path='predictions_plot.png'):
        """Plot actual vs predicted values"""
        if self.model is None or self.scaler is None:
            print("Model not trained yet!")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title('Actual vs Predicted House Prices', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.show()
        print(f"Plot saved to {save_path}")
    
    def save_model(self, model_path='house_price_model.pkl', scaler_path='scaler.pkl'):
        """Save model and scaler to disk"""
        if self.model is None or self.scaler is None:
            print("No model to save!")
            return
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Also save feature columns
        with open('feature_columns.txt', 'w') as f:
            for col in self.feature_columns:
                f.write(f"{col}\n")
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Feature columns saved to feature_columns.txt")

def main():
    # Initialize model
    model = HousePriceModel()
    
    # Load data
    X, y = model.load_and_prepare_data()
    
    # Split data for plotting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    metrics, feature_importance = model.train(X, y, use_regularization=False)
    
    # Plot predictions
    model.plot_predictions(X_test, y_test)
    
    # Create correlation heatmap
    data = pd.read_csv('house_data.csv')
    plt.figure(figsize=(10, 8))
    correlation = data[model.feature_columns + ['price']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlations', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=100)
    plt.show()
    
    # Save model
    model.save_model()
    
    return model

if __name__ == "__main__":
    model = main()
    print("\n✅ Model training completed successfully!")