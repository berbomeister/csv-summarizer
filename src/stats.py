import pandas as pd
import numpy as np

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Returns basic statistics about the dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isna().sum().sum(),
        "duplicated_rows": df.duplicated().sum()
    }

def get_column_stats(df: pd.DataFrame, column: str) -> dict:
    """Returns statistics for a specific column."""
    if column not in df.columns:
        return {}
    
    col_data = df[column]
    stats = {
        "dtype": str(col_data.dtype),
        "missing": col_data.isna().sum(),
        "num_distinct": col_data.nunique(),
    }
    
    if pd.api.types.is_numeric_dtype(col_data):
        stats["min"] = col_data.min()
        stats["max"] = col_data.max()
        stats["mean"] = col_data.mean()
        stats["std"] = col_data.std()
    else:
        # Categorical stats
        pass
        
    return stats

def get_column_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame containing summary statistics for all columns."""
    rows = []
    for col in df.columns:
        col_data = df[col]
        is_num = pd.api.types.is_numeric_dtype(col_data)
        
        row = {
            "Column": col,
            "Type": str(col_data.dtype),
            "Missing": col_data.isna().sum(),
            "Unique": col_data.nunique(),
        }
        
        if is_num:
            row["Min"] = f"{col_data.min():.2f}"
            row["Max"] = f"{col_data.max():.2f}"
            row["Mean"] = f"{col_data.mean():.2f}"
        else:
            row["Min"] = "-"
            row["Max"] = "-"
            row["Mean"] = "-"
            
        rows.append(row)
        
    return pd.DataFrame(rows)
