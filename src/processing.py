import pandas as pd
import numpy as np

def get_missing_columns(df: pd.DataFrame) -> list[str]:
    """Returns a list of columns that have missing values."""
    return df.columns[df.isna().any()].tolist()

def impute_data(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    """
    Imputes missing values in the specified column using the given strategy.
    Strategies:
    - 'drop_rows': Removes rows with missing values in this column.
    - 'mean': Fills with mean (Numerical only).
    - 'median': Fills with median (Numerical only).
    - 'zero': Fills with 0 (Numerical only).
    - 'mode': Fills with most frequent value.
    - 'missing_label': Fills with 'Missing' (Categorical only).
    """
    df = df.copy()
    
    if column not in df.columns:
        return df
        
    if strategy == "drop_rows":
        df = df.dropna(subset=[column])
        return df
        
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    
    if strategy == "mean" and is_numeric:
        val = df[column].mean()
        df[column] = df[column].fillna(val)
    elif strategy == "median" and is_numeric:
        val = df[column].median()
        df[column] = df[column].fillna(val)
    elif strategy == "zero" and is_numeric:
        df[column] = df[column].fillna(0)
    elif strategy == "mode":
        # Check if mode exists (not empty)
        if not df[column].mode().empty:
            val = df[column].mode()[0]
            df[column] = df[column].fillna(val)
    elif strategy == "missing_label" and not is_numeric:
        df[column] = df[column].fillna("Missing")
        
    return df
