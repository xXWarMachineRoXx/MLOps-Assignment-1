import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.data_preprocessing import clean_data

def test_clean_data():
    df = pd.DataFrame({
        'age': [50, 60, 70],
        'sex': [1, 0, 1],
        'cp': [0, 1, 2],
        'trestbps': [120, 130, 140],
        'chol': [200, 220, 240],
        'fbs': [0, 1, 0],
        'restecg': [0, 1, 0],
        'thalach': [150, 160, 170],
        'exang': [0, 1, 0],
        'oldpeak': [1.0, 2.0, 1.5],
        'slope': [0, 1, 2],
        'ca': [0, 1, 2],
        'thal': [1, 2, 3],
        'target': [0, 2, 1]
    })
    
    cleaned = clean_data(df.copy())
    assert cleaned['target'].max() == 1
    assert cleaned['target'].min() == 0
