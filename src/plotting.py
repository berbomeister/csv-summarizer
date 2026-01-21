import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_numerical_index(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.plot(df.index, df[col])
    ax.set_title(f"Index Plot of {col}")
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    return fig

def plot_numerical_boxplot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    return fig

def plot_numerical_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Histogram of {col}")
    return fig

def plot_categorical_bar(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    # Count distinct values
    counts = df[col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f"Barplot of {col}")
    ax.tick_params(axis='x', rotation=45)
    return fig

def plot_scatterplot(df: pd.DataFrame, col1: str, col2: str, hue: str = None):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, hue=hue, ax=ax)
    title = f"Scatterplot: {col1} vs {col2}"
    if hue:
        title += f" (colored by {hue})"
    ax.set_title(title)
    return fig

def plot_grouped_boxplot(df: pd.DataFrame, num_col: str, cat_col: str):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
    ax.set_title(f"Boxplot of {num_col} by {cat_col}")
    ax.tick_params(axis='x', rotation=45)
    return fig

def plot_grouped_density(df: pd.DataFrame, num_col: str, cat_col: str):
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=num_col, hue=cat_col, fill=True, ax=ax)
    ax.set_title(f"Density Plot of {num_col} by {cat_col}")
    return fig

def plot_categorical_heatmap(df: pd.DataFrame, col1: str, col2: str):
    fig, ax = plt.subplots()
    ct = pd.crosstab(df[col1], df[col2])
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    ax.set_title(f"Contingency Heatmap: {col1} vs {col2}")
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        ax.text(0.5, 0.5, "No numeric columns for correlation matrix", 
                ha='center', va='center')
        return fig
        
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    return fig

def plot_variable_importance(importance_df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title("Variable Importance")
    return fig
