import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
import warnings
from src.plotting import (
    plot_numerical_index, plot_numerical_boxplot, plot_numerical_histogram,
    plot_categorical_bar, plot_scatterplot, plot_grouped_boxplot,
    plot_grouped_density, plot_categorical_heatmap, plot_correlation_matrix
)

@pytest.mark.filterwarnings("ignore:vert:PendingDeprecationWarning")
def test_plots():
    df = pd.DataFrame({
        'num1': np.random.rand(50),
        'num2': np.random.rand(50),
        'cat1': np.random.choice(['A', 'B'], 50),
        'cat2': np.random.choice(['X', 'Y'], 50)
    })

    # Just ensure they run without error and return a Figure
    assert isinstance(plot_numerical_index(df, 'num1'), plt.Figure)
    assert isinstance(plot_numerical_boxplot(df, 'num1'), plt.Figure)
    assert isinstance(plot_numerical_histogram(df, 'num1'), plt.Figure)
    assert isinstance(plot_categorical_bar(df, 'cat1'), plt.Figure)
    assert isinstance(plot_scatterplot(df, 'num1', 'num2'), plt.Figure)
    assert isinstance(plot_grouped_boxplot(df, 'num1', 'cat1'), plt.Figure)
    assert isinstance(plot_grouped_density(df, 'num1', 'cat1'), plt.Figure)
    assert isinstance(plot_categorical_heatmap(df, 'cat1', 'cat2'), plt.Figure)
    assert isinstance(plot_correlation_matrix(df), plt.Figure)
    plt.close('all')
