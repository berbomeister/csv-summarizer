import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

def train_linear_model(df: pd.DataFrame, target: str, features: list[str], test_size: float):
    X = df[features]
    y = df[target]
    
    # Handle missing values simply (drop rows with missing in selected cols)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[features]
    y = data[target]
    
    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "Test MAE": mean_absolute_error(y_test, y_pred_test),
        "Train R2": r2_score(y_train, y_pred_train),
        "Test R2": r2_score(y_test, y_pred_test)
    }
    
    # Linear Regression coefficients as importance
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.abs(model.coef_)
    }).sort_values(by="Importance", ascending=False)
    
    return metrics, importance

def train_adaboost_model(df: pd.DataFrame, target: str, features: list[str], test_size: float):
    X = df[features]
    y = df[target]
    
    # Handle missing values
    data = pd.concat([X, y], axis=1).dropna()
    X = data[features]
    y = data[target]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    
    # Encode target if not numeric
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Train Precision": precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Test Precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "Train F1-Score": f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Test F1-Score": f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    }
    
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return metrics, importance
