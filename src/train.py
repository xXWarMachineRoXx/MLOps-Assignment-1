import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os

def prepare_features(df):
    """Feature engineering and scaling"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models\\scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with MLflow tracking"""
    with mlflow.start_run(run_name="Logistic_Regression"):
        params = {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
        mlflow.log_params(params)
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        joblib.dump(model, 'models\\logistic_regression.pkl')
        
        print(f"✓ Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with MLflow tracking"""
    with mlflow.start_run(run_name="Random_Forest"):
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        joblib.dump(model, 'models\\random_forest.pkl')
        
        print(f"✓ Random Forest - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("heart_disease_prediction")
    
    df = pd.read_csv('data\\heart_disease_clean.csv')
    
    os.makedirs('models', exist_ok=True)
    
    X_train, X_test, y_train, y_test, scaler = prepare_features(df)
    
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    
    best_model_name = 'random_forest' if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else 'logistic_regression'
    print(f"\n✓ Best model: {best_model_name}")
    
    with open('models\\best_model.txt', 'w') as f:
        f.write(best_model_name)
