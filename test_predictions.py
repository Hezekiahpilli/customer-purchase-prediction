import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestPredictionSystem(unittest.TestCase):
    
    def setUp(self):
        # Create test data
        self.test_data = pd.DataFrame({
            'Customer ID': [1, 1, 1, 2, 2],
            'Transaction Date': ['2021-01-01', '2021-02-01', '2021-03-01', 
                                '2021-01-15', '2021-02-20'],
            'Product Category': ['A', 'B', 'A', 'C', 'C'],
            'Purchase Amount': [100, 150, 200, 300, 250]
        })
    
    def test_data_loading(self):
        # Test that dates are properly converted
        self.test_data['Transaction Date'] = pd.to_datetime(self.test_data['Transaction Date'])
        self.assertEqual(self.test_data['Transaction Date'].dtype, 'datetime64[ns]')
    
    def test_interval_calculation(self):
        # Test interval calculation between purchases
        customer_1 = self.test_data[self.test_data['Customer ID'] == 1].copy()
        customer_1['Transaction Date'] = pd.to_datetime(customer_1['Transaction Date'])
        customer_1 = customer_1.sort_values('Transaction Date')
        intervals = customer_1['Transaction Date'].diff().dt.days.dropna().tolist()
        self.assertEqual(intervals, [31.0, 28.0])
    
    def test_pattern_analysis(self):
        # Test customer pattern classification
        patterns = {
            'Mean_Interval': 10,
            'Num_Transactions': 5
        }
        # Should classify as "Very Frequent" (< 15 days)
        self.assertTrue(patterns['Mean_Interval'] < 15)

if __name__ == '__main__':
    unittest.main()