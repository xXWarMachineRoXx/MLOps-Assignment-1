import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports" / "figures"

def download_data():
    """Download Heart Disease UCI dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df.to_csv(DATA_DIR / "heart_disease_raw.csv", index=False)
    print(f"✓ Downloaded dataset: {df.shape}")
    return df

def clean_data(df):
    """Clean and preprocess data"""
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    
    print(f"✓ Cleaned dataset: {df.shape}")
    print(f"✓ Class distribution:\n{df['target'].value_counts()}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "heart_disease_clean.csv", index=False)
    return df

def perform_eda(df):
    """Perform exploratory data analysis"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    for idx, col in enumerate(numeric_cols):
        row, col_idx = idx // 3, idx % 3
        sns.histplot(data=df, x=col, hue='target', kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} Distribution by Target')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class balance
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution')
    plt.xlabel('Target (0: No Disease, 1: Disease)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig(REPORTS_DIR / "class_balance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ EDA visualizations saved to {REPORTS_DIR}")

if __name__ == "__main__":
    df = download_data()
    df = clean_data(df)
    perform_eda(df)
