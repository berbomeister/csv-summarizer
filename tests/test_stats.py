import pandas as pd
import numpy as np
from src.stats import get_dataset_stats, get_column_stats

def test_get_dataset_stats():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    stats = get_dataset_stats(df)
    assert stats['rows'] == 3
    assert stats['columns'] == 2
    assert stats['missing_values'] == 0

def test_get_column_stats_numeric():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    stats = get_column_stats(df, 'A')
    assert stats['min'] == 1
    assert stats['max'] == 5
    assert stats['mean'] == 3

def test_get_column_stats_categorical():
    df = pd.DataFrame({'B': ['a', 'b', 'a', 'c']})
    stats = get_column_stats(df, 'B')
    assert stats['num_distinct'] == 3
