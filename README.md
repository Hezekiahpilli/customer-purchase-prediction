# Customer Purchase Prediction System

A machine learning system that predicts when customers will make their next purchase based on historical transaction data.

## Overview

This system analyzes customer purchase patterns and uses machine learning models to predict:
- When each customer will make their next purchase
- Customer segmentation based on buying behavior
- Confidence levels for each prediction
- Business insights and recommendations

## Features

- **Data Analysis**: Comprehensive customer behavior pattern analysis
- **Feature Engineering**: Automatic creation of predictive features from transaction history
- **Multiple ML Models**: Compares Linear Regression, Random Forest, Gradient Boosting, and XGBoost
- **Customer Segmentation**: Classifies customers into behavioral segments (New, Very Frequent, Frequent, Regular, Occasional, Rare)
- **Detailed Reports**: Generates analysis reports with business recommendations
- **Visualizations**: Creates charts showing predictions, segments, and model performance
- **Test Mode**: Includes sample data generation for testing

## Requirements

### Core Requirements (Minimum)
- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

### Recommended (Full Functionality)
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### Optional (Better Performance)
- xgboost >= 1.5.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hezekiahpilli/customer-purchase-prediction.git
cd customer-purchase-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### With Your Own Data

Prepare two CSV files with the following structure:

**train.csv** and **test.csv**:
```csv
Customer ID,Transaction Date,Product Category,Purchase Amount
1001,2021-01-15,Electronics,150.50
1001,2021-02-20,Clothing,75.00
```

Run the prediction:
```bash
python complete_implementation.py --train train.csv --test test.csv
```

### Test Mode (Sample Data)

Run with automatically generated sample data:
```bash
python complete_implementation.py --test-mode
```

### Command Line Options

```bash
python complete_implementation.py [OPTIONS]

Options:
  --train PATH       Path to training data CSV (default: train.csv)
  --test PATH        Path to test data CSV (default: test.csv)
  --output DIR       Output directory for results (default: current directory)
  --test-mode        Run with automatically generated sample data
```

## Output Files

The system generates three output files:

1. **customer_predictions.csv**: Predictions for all customers
   - Customer ID
   - Last Purchase Date
   - Predicted Days Until Next Purchase
   - Predicted Next Purchase Date
   - Confidence (%)
   - Customer Segment
   - Historical Purchases
   - Average Purchase Amount

2. **analysis_report.txt**: Detailed analysis report including:
   - Executive summary
   - Customer segmentation breakdown
   - Model performance metrics
   - Business recommendations

3. **analysis_plots.png**: Visualization charts showing:
   - Distribution of predictions
   - Customer segment pie chart
   - Model performance comparison
   - Confidence distribution

## How It Works

### 1. Data Loading
- Loads training and test data
- Combines for complete customer history
- Converts dates and validates data

### 2. Pattern Analysis
- Analyzes purchase intervals for each customer
- Calculates statistics (mean, median, weighted average)
- Classifies customers into segments

### 3. Feature Engineering
- Creates predictive features from transaction sequences
- Includes temporal, behavioral, and historical features
- Generates training samples for ML models

### 4. Model Training
- Trains multiple ML models (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
- Evaluates performance using MAE, RMSE, and R² metrics
- Selects best performing model

### 5. Predictions
- Generates predictions for all customers
- Applies smart heuristics based on purchase patterns
- Calculates confidence levels
- Bounds predictions between 7-180 days

### 6. Results & Visualization
- Saves predictions and analysis reports
- Creates visualization charts
- Provides business recommendations

## Customer Segments

- **New**: 1 or fewer transactions
- **Very Frequent**: Average purchase interval < 15 days
- **Frequent**: Average purchase interval < 30 days
- **Regular**: Average purchase interval < 60 days
- **Occasional**: Average purchase interval < 90 days
- **Rare**: Average purchase interval ≥ 90 days

## Model Performance

The system compares multiple models and automatically selects the best performer:
- **Linear Regression**: Fast baseline model
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Advanced boosting algorithm
- **XGBoost** (optional): High-performance gradient boosting

Performance is evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error in days
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (R-squared): Proportion of variance explained

## Example Output

```
Sample Predictions (first 5):
Customer ID  Last Purchase Date  Predicted Days Until Next Purchase  Confidence (%)  Customer Segment
1001        2023-06-15          28                                   90             Frequent
1002        2023-06-20          45                                   70             Regular
1003        2023-06-10          15                                   90             Very Frequent
```

## Business Applications

- **Targeted Marketing**: Identify customers likely to purchase soon
- **Retention Campaigns**: Re-engage customers with longer intervals
- **Inventory Planning**: Predict demand based on purchase patterns
- **Customer Lifetime Value**: Estimate future purchase frequency
- **Loyalty Programs**: Identify and reward frequent customers

## License

This project is open source and available for use.

## Contact

For questions or issues, please open an issue on GitHub.

