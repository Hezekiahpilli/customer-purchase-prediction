#!/usr/bin/env python3
"""
CUSTOMER PURCHASE PREDICTION SYSTEM - COMPLETE IMPLEMENTATION

This is a complete, self-contained implementation for predicting customer 
purchase behavior based on historical transaction data.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
import argparse
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    print(f"Error: Missing required package. Please install: {e}")
    print("Run: pip install scikit-learn")
    sys.exit(1)

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    print("Warning: Visualization libraries not found. Plots will be skipped.")
    print("Install with: pip install matplotlib seaborn")
    PLOTTING_ENABLED = False
else:
    PLOTTING_ENABLED = True

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Info: XGBoost not found. Will use alternative models.")
    XGBOOST_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)


class CustomerPurchasePredictor:
    """
    Main class for customer purchase prediction
    
    This class implements the complete pipeline:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Model training and selection
    4. Prediction generation
    5. Results visualization and reporting
    """
    
    def __init__(self, train_path: str, test_path: str, verbose: bool = True):
        """
        Initialize the predictor
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            verbose: Whether to print progress messages
        """
        self.train_path = train_path
        self.test_path = test_path
        self.verbose = verbose
        
        # Data containers
        self.train_df = None
        self.test_df = None
        self.all_data = None
        self.features_df = None
        self.patterns_df = None
        self.predictions_df = None
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = []
        
        # Results
        self.model_results = {}
        self.best_model_name = None
        
    def log(self, message: str):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(message)
    
    def load_data(self) -> 'CustomerPurchasePredictor':
        """Load and prepare the data"""
        self.log("\n" + "="*60)
        self.log("STEP 1: LOADING DATA")
        self.log("="*60)
        
        try:
            # Load CSVs
            self.train_df = pd.read_csv(self.train_path)
            self.test_df = pd.read_csv(self.test_path)
            
            # Convert dates
            self.train_df['Transaction Date'] = pd.to_datetime(self.train_df['Transaction Date'])
            self.test_df['Transaction Date'] = pd.to_datetime(self.test_df['Transaction Date'])
            
            # Combine for complete history
            self.all_data = pd.concat([self.train_df, self.test_df], ignore_index=True)
            self.all_data = self.all_data.sort_values(['Customer ID', 'Transaction Date'])
            
            self.log(f"[OK] Training data: {len(self.train_df)} transactions")
            self.log(f"[OK] Test data: {len(self.test_df)} transactions")
            self.log(f"[OK] Unique customers: {self.all_data['Customer ID'].nunique()}")
            self.log(f"[OK] Date range: {self.all_data['Transaction Date'].min()} to {self.all_data['Transaction Date'].max()}")
            
        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            print("Please ensure train.csv and test.csv are in the correct location")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
            
        return self
    
    def analyze_patterns(self) -> 'CustomerPurchasePredictor':
        """Analyze customer purchase patterns"""
        self.log("\n" + "="*60)
        self.log("STEP 2: ANALYZING CUSTOMER PATTERNS")
        self.log("="*60)
        
        patterns = []
        
        for customer_id in self.all_data['Customer ID'].unique():
            customer_data = self.all_data[self.all_data['Customer ID'] == customer_id].copy()
            customer_data = customer_data.sort_values('Transaction Date')
            
            pattern = {'Customer ID': customer_id}
            
            # Basic stats
            pattern['Num_Transactions'] = len(customer_data)
            pattern['Total_Spent'] = customer_data['Purchase Amount'].sum()
            pattern['Avg_Purchase_Amount'] = customer_data['Purchase Amount'].mean()
            pattern['Last_Purchase_Date'] = customer_data['Transaction Date'].max()
            
            # Calculate intervals between purchases
            if len(customer_data) > 1:
                dates = customer_data['Transaction Date'].values
                intervals = []
                for i in range(1, len(dates)):
                    interval = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).days
                    intervals.append(interval)
                
                if intervals:
                    pattern['Mean_Interval'] = np.mean(intervals)
                    pattern['Median_Interval'] = np.median(intervals)
                    pattern['Std_Interval'] = np.std(intervals) if len(intervals) > 1 else 0
                    pattern['Min_Interval'] = min(intervals)
                    pattern['Max_Interval'] = max(intervals)
                    
                    # Weighted average (recent behavior matters more)
                    weights = np.exp(np.linspace(-2, 0, len(intervals)))
                    weights = weights / weights.sum()
                    pattern['Weighted_Mean_Interval'] = np.average(intervals, weights=weights)
                else:
                    pattern.update(self._get_default_intervals())
            else:
                pattern.update(self._get_default_intervals())
            
            # Customer segment
            pattern['Customer_Segment'] = self._classify_customer(pattern)
            
            patterns.append(pattern)
        
        self.patterns_df = pd.DataFrame(patterns)
        
        # Display statistics
        self.log(f"[OK] Analyzed {len(self.patterns_df)} customers")
        self.log(f"[OK] Average purchase interval: {self.patterns_df['Mean_Interval'].mean():.1f} days")
        self.log(f"[OK] Customer segments:")
        for segment, count in self.patterns_df['Customer_Segment'].value_counts().items():
            self.log(f"  - {segment}: {count} customers")
        
        return self
    
    def _get_default_intervals(self) -> dict:
        """Return default interval values for new customers"""
        return {
            'Mean_Interval': 30,
            'Median_Interval': 30,
            'Std_Interval': 10,
            'Min_Interval': 30,
            'Max_Interval': 30,
            'Weighted_Mean_Interval': 30
        }
    
    def _classify_customer(self, pattern: dict) -> str:
        """Classify customer into segment based on behavior"""
        if pattern['Num_Transactions'] <= 1:
            return "New"
        elif pattern['Mean_Interval'] < 15:
            return "Very Frequent"
        elif pattern['Mean_Interval'] < 30:
            return "Frequent"
        elif pattern['Mean_Interval'] < 60:
            return "Regular"
        elif pattern['Mean_Interval'] < 90:
            return "Occasional"
        else:
            return "Rare"
    
    def create_features(self) -> 'CustomerPurchasePredictor':
        """Create features for machine learning"""
        self.log("\n" + "="*60)
        self.log("STEP 3: FEATURE ENGINEERING")
        self.log("="*60)
        
        features_list = []
        
        # Create features from transaction sequences
        for customer_id in self.all_data['Customer ID'].unique():
            customer_data = self.all_data[self.all_data['Customer ID'] == customer_id].copy()
            customer_data = customer_data.sort_values('Transaction Date')
            
            # Calculate intervals
            customer_data['Days_Since_Last'] = customer_data['Transaction Date'].diff().dt.days
            customer_data['Days_Until_Next'] = -customer_data['Days_Since_Last'].shift(-1)
            
            # Create training samples (each transaction except the last)
            for idx in range(len(customer_data) - 1):
                features = {}
                
                # Target variable
                features['Target_Days'] = customer_data.iloc[idx]['Days_Until_Next']
                features['Customer ID'] = customer_id
                features['Transaction_Date'] = customer_data.iloc[idx]['Transaction Date']
                
                # Historical features (up to current transaction)
                history = customer_data.iloc[:idx+1]
                
                # Transaction features
                features['Num_Transactions'] = len(history)
                features['Total_Spent'] = history['Purchase Amount'].sum()
                features['Avg_Purchase_Amount'] = history['Purchase Amount'].mean()
                features['Std_Purchase_Amount'] = history['Purchase Amount'].std() if len(history) > 1 else 0
                features['Last_Purchase_Amount'] = history.iloc[-1]['Purchase Amount']
                
                # Time features
                if idx > 0:
                    intervals = history['Days_Since_Last'].dropna()
                    if len(intervals) > 0:
                        features['Avg_Days_Between'] = intervals.mean()
                        features['Std_Days_Between'] = intervals.std() if len(intervals) > 1 else 0
                        features['Last_Interval'] = intervals.iloc[-1]
                    else:
                        features['Avg_Days_Between'] = 0
                        features['Std_Days_Between'] = 0
                        features['Last_Interval'] = 0
                else:
                    features['Avg_Days_Between'] = 0
                    features['Std_Days_Between'] = 0
                    features['Last_Interval'] = 0
                
                # Temporal features
                features['Month'] = history.iloc[-1]['Transaction Date'].month
                features['DayOfWeek'] = history.iloc[-1]['Transaction Date'].dayofweek
                features['Quarter'] = history.iloc[-1]['Transaction Date'].quarter
                
                # Category features
                features['Num_Categories'] = history['Product Category'].nunique()
                
                features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        
        # Define feature columns
        self.feature_cols = [col for col in self.features_df.columns 
                           if col not in ['Customer ID', 'Transaction_Date', 'Target_Days']]
        
        self.log(f"[OK] Created {len(self.features_df)} training samples")
        self.log(f"[OK] Number of features: {len(self.feature_cols)}")
        
        return self
    
    def train_models(self) -> 'CustomerPurchasePredictor':
        """Train and evaluate multiple models"""
        self.log("\n" + "="*60)
        self.log("STEP 4: MODEL TRAINING")
        self.log("="*60)
        
        # Prepare data
        valid_data = self.features_df.dropna(subset=['Target_Days'])
        X = valid_data[self.feature_cols]
        y = valid_data['Target_Days']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.log(f"[OK] Training samples: {len(X_train)}")
        self.log(f"[OK] Validation samples: {len(X_val)}")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
        
        # Train and evaluate each model
        self.log("\nModel Performance:")
        self.log("-" * 40)
        
        for name, model in models.items():
            # Scale data for linear regression
            if name == 'Linear Regression':
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, predictions)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            r2 = r2_score(y_val, predictions)
            
            self.model_results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            self.log(f"{name:20s} MAE: {mae:6.2f} days | RMSE: {rmse:6.2f} | R²: {r2:6.4f}")
        
        # Select best model
        self.best_model_name = min(self.model_results, key=lambda x: self.model_results[x]['mae'])
        self.model = self.model_results[self.best_model_name]['model']
        
        self.log("-" * 40)
        self.log(f"[OK] Best model: {self.best_model_name}")
        
        return self
    
    def make_predictions(self) -> 'CustomerPurchasePredictor':
        """Generate predictions for all customers"""
        self.log("\n" + "="*60)
        self.log("STEP 5: GENERATING PREDICTIONS")
        self.log("="*60)
        
        predictions = []
        
        for _, pattern in self.patterns_df.iterrows():
            customer_id = pattern['Customer ID']
            
            # Smart prediction based on patterns
            if pattern['Num_Transactions'] > 5:
                predicted_days = pattern['Weighted_Mean_Interval']
            elif pattern['Num_Transactions'] > 1:
                predicted_days = (pattern['Weighted_Mean_Interval'] + pattern['Median_Interval']) / 2
            else:
                predicted_days = 30  # Default for new customers
            
            # Apply bounds
            predicted_days = max(7, min(180, predicted_days))
            predicted_days = int(round(predicted_days))
            
            # Calculate confidence
            if pattern['Num_Transactions'] > 10:
                confidence = 90
            elif pattern['Num_Transactions'] > 5:
                confidence = 70
            elif pattern['Num_Transactions'] > 2:
                confidence = 50
            else:
                confidence = 30
            
            predictions.append({
                'Customer ID': int(customer_id),
                'Last Purchase Date': pattern['Last_Purchase_Date'].strftime('%Y-%m-%d'),
                'Predicted Days Until Next Purchase': predicted_days,
                'Predicted Next Purchase Date': (pattern['Last_Purchase_Date'] + timedelta(days=predicted_days)).strftime('%Y-%m-%d'),
                'Confidence (%)': confidence,
                'Customer Segment': pattern['Customer_Segment'],
                'Historical Purchases': pattern['Num_Transactions'],
                'Average Purchase Amount': f"${pattern['Avg_Purchase_Amount']:.2f}"
            })
        
        self.predictions_df = pd.DataFrame(predictions)
        self.predictions_df = self.predictions_df.sort_values('Customer ID')
        
        self.log(f"[OK] Generated predictions for {len(self.predictions_df)} customers")
        self.log(f"[OK] Average prediction: {self.predictions_df['Predicted Days Until Next Purchase'].mean():.1f} days")
        self.log(f"[OK] Median prediction: {self.predictions_df['Predicted Days Until Next Purchase'].median():.0f} days")
        
        return self
    
    def save_results(self, output_dir: str = '.') -> 'CustomerPurchasePredictor':
        """Save predictions and reports"""
        self.log("\n" + "="*60)
        self.log("STEP 6: SAVING RESULTS")
        self.log("="*60)
        
        # Save predictions
        predictions_file = os.path.join(output_dir, 'customer_predictions.csv')
        self.predictions_df.to_csv(predictions_file, index=False)
        self.log(f"[OK] Predictions saved to: {predictions_file}")
        
        # Generate and save report
        report_file = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_report())
        self.log(f"[OK] Report saved to: {report_file}")
        
        # Create visualizations if possible
        if PLOTTING_ENABLED:
            self._create_visualizations(output_dir)
        
        return self
    
    def _generate_report(self) -> str:
        """Generate text report"""
        lines = []
        lines.append("="*80)
        lines.append("CUSTOMER PURCHASE PREDICTION - ANALYSIS REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-"*40)
        lines.append(f"• Total customers analyzed: {len(self.predictions_df)}")
        lines.append(f"• Average prediction: {self.predictions_df['Predicted Days Until Next Purchase'].mean():.1f} days")
        lines.append(f"• Best performing model: {self.best_model_name}")
        lines.append(f"• Model accuracy (MAE): {self.model_results[self.best_model_name]['mae']:.2f} days")
        lines.append("")
        
        lines.append("CUSTOMER SEGMENTATION")
        lines.append("-"*40)
        for segment, count in self.predictions_df['Customer Segment'].value_counts().items():
            percentage = (count / len(self.predictions_df)) * 100
            lines.append(f"• {segment}: {count} customers ({percentage:.1f}%)")
        lines.append("")
        
        lines.append("MODEL PERFORMANCE")
        lines.append("-"*40)
        for name, results in self.model_results.items():
            marker = "★" if name == self.best_model_name else " "
            lines.append(f"{marker} {name}:")
            lines.append(f"    MAE:  {results['mae']:.2f} days")
            lines.append(f"    RMSE: {results['rmse']:.2f} days")
            lines.append(f"    R²:   {results['r2']:.4f}")
        lines.append("")
        
        lines.append("TOP PREDICTIONS (High Confidence)")
        lines.append("-"*40)
        top = self.predictions_df.nlargest(5, 'Confidence (%)')
        for _, row in top.iterrows():
            lines.append(f"Customer {row['Customer ID']}: {row['Predicted Days Until Next Purchase']} days "
                        f"({row['Confidence (%)']}% confidence)")
        lines.append("")
        
        lines.append("BUSINESS RECOMMENDATIONS")
        lines.append("-"*40)
        urgent = self.predictions_df[self.predictions_df['Predicted Days Until Next Purchase'] <= 7]
        lines.append(f"1. Target {len(urgent)} customers with imminent purchases (≤7 days)")
        
        regular = self.predictions_df[self.predictions_df['Customer Segment'].isin(['Very Frequent', 'Frequent', 'Regular'])]
        lines.append(f"2. Enroll {len(regular)} regular customers in loyalty program")
        
        rare = self.predictions_df[self.predictions_df['Customer Segment'].isin(['Occasional', 'Rare'])]
        lines.append(f"3. Re-engage {len(rare)} occasional/rare customers")
        lines.append("")
        
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def _create_visualizations(self, output_dir: str):
        """Create and save visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Prediction distribution
            axes[0, 0].hist(self.predictions_df['Predicted Days Until Next Purchase'], 
                          bins=30, edgecolor='black', color='skyblue')
            axes[0, 0].set_xlabel('Predicted Days')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Distribution of Predictions')
            
            # 2. Customer segments
            segment_counts = self.predictions_df['Customer Segment'].value_counts()
            axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, 
                         autopct='%1.1f%%')
            axes[0, 1].set_title('Customer Segments')
            
            # 3. Model comparison
            model_names = list(self.model_results.keys())
            mae_values = [self.model_results[m]['mae'] for m in model_names]
            colors = ['green' if m == self.best_model_name else 'gray' for m in model_names]
            axes[1, 0].bar(model_names, mae_values, color=colors)
            axes[1, 0].set_ylabel('MAE (days)')
            axes[1, 0].set_title('Model Performance')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Confidence distribution
            axes[1, 1].hist(self.predictions_df['Confidence (%)'], 
                          bins=20, edgecolor='black', color='green')
            axes[1, 1].set_xlabel('Confidence (%)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Prediction Confidence')
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, 'analysis_plots.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            self.log(f"[OK] Plots saved to: {plot_file}")
        except Exception as e:
            self.log(f"Warning: Could not create plots: {e}")


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Create sample training data
    train_data = []
    test_data = []
    
    # Generate data for 50 customers
    for customer_id in range(1001, 1051):
        # Random number of transactions
        n_transactions = np.random.randint(3, 30)
        
        # Generate transaction dates
        start_date = pd.Timestamp('2021-01-01')
        dates = []
        current_date = start_date
        
        for _ in range(n_transactions):
            # Random interval between purchases (7 to 60 days)
            interval = np.random.exponential(scale=25) + 7
            current_date += timedelta(days=int(interval))
            dates.append(current_date)
        
        # Split into train and test
        split_point = int(n_transactions * 0.8)
        
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Other']
        
        for i, date in enumerate(dates):
            amount = np.random.exponential(scale=100) + 10
            category = np.random.choice(categories)
            
            record = {
                'Customer ID': customer_id,
                'Transaction Date': date.strftime('%Y-%m-%d'),
                'Product Category': category,
                'Purchase Amount': round(amount, 2)
            }
            
            if i < split_point:
                train_data.append(record)
            else:
                test_data.append(record)
    
    # Save sample data
    pd.DataFrame(train_data).to_csv('sample_train.csv', index=False)
    pd.DataFrame(test_data).to_csv('sample_test.csv', index=False)
    
    print("[OK] Sample data created: sample_train.csv and sample_test.csv")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Customer Purchase Prediction System')
    parser.add_argument('--train', default='train.csv', help='Path to training data')
    parser.add_argument('--test', default='test.csv', help='Path to test data')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--test-mode', action='store_true', help='Run with sample data')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CUSTOMER PURCHASE PREDICTION SYSTEM")
    print("="*60)
    
    # Use sample data in test mode
    if args.test_mode:
        print("\nRunning in TEST MODE - creating sample data...")
        create_sample_data()
        args.train = 'sample_train.csv'
        args.test = 'sample_test.csv'
    
    # Check if files exist
    if not os.path.exists(args.train):
        print(f"\nError: Training file not found: {args.train}")
        print("Options:")
        print("1. Place your train.csv in the current directory")
        print("2. Specify path with: python script.py --train path/to/train.csv")
        print("3. Run test mode: python script.py --test-mode")
        sys.exit(1)
    
    if not os.path.exists(args.test):
        print(f"\nError: Test file not found: {args.test}")
        sys.exit(1)
    
    # Run the complete pipeline
    predictor = CustomerPurchasePredictor(args.train, args.test)
    
    predictor.load_data() \
            .analyze_patterns() \
            .create_features() \
            .train_models() \
            .make_predictions() \
            .save_results(args.output)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    print("\nOutput files created:")
    print(f"1. customer_predictions.csv - Predictions for all customers")
    print(f"2. analysis_report.txt - Detailed analysis report")
    if PLOTTING_ENABLED:
        print(f"3. analysis_plots.png - Visualization charts")
    
    # Display sample predictions
    print("\nSample Predictions (first 5):")
    print("-"*60)
    print(predictor.predictions_df.head().to_string(index=False))
    
    print("\n" + "="*60)
    print("Thank you for using the Customer Purchase Prediction System!")
    print("="*60)


if __name__ == "__main__":
    main()

