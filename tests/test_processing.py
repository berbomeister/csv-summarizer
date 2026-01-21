import pandas as pd
import numpy as np
from src.processing import impute_data, get_missing_columns

def test_get_missing_columns():
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': [1, 2, 3],
        'C': [np.nan, 'b', 'c']
    })
    cols = get_missing_columns(df)
    assert 'A' in cols
    assert 'C' in cols
    assert 'B' not in cols

def test_impute_drop_rows():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    df_clean = impute_data(df, 'A', 'drop_rows')
    assert len(df_clean) == 2
    assert df_clean['A'].isna().sum() == 0

def test_impute_mean():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    df_clean = impute_data(df, 'A', 'mean')
    assert df_clean['A'].iloc[1] == 2.0
    assert df_clean['A'].isna().sum() == 0

def test_impute_mode():
    df = pd.DataFrame({'A': ['a', 'a', np.nan, 'b']})
    df_clean = impute_data(df, 'A', 'mode')
    assert df_clean['A'].iloc[2] == 'a'
    assert df_clean['A'].isna().sum() == 0

def test_impute_missing_label():
    df = pd.DataFrame({'A': ['a', np.nan, 'b']})
    df_clean = impute_data(df, 'A', 'missing_label')
    assert df_clean['A'].iloc[1] == 'Missing'
    assert df_clean['A'].isna().sum() == 0
