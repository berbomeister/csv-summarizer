# CSV Summarizer App

This project contains all the necessary code to create and run a Shiny for Python application that allows users to upload CSV files, view statistics, create visualizations, and train simple machine learning models.

## Functionality

### 1. Welcome Page
Provides an overview of the application and instructions on how to navigate the tabs.

### 2. Data & Stats
- **Data Preview**: Shows the raw data from the uploaded CSV (defaults to the Iris dataset).
- **Statistics**: Displays summary statistics for the entire dataset and a detailed table for all columns (Type, Missing Values, Unique Counts, Min/Max/Mean).
- **Variable Type Management**: Allows users to override inferred column types (e.g., converting a numeric ID column to Categorical).
- **Missing Data Handling**: Provides tools to handle missing values by dropping rows or imputing values (Mean, Median, Zero, Mode, or a "Missing" label).

### 3. Visualizations
- **Correlation Matrix**: A heatmap showing correlations between numerical variables.
- **Dynamic Plotting**: Automatically generates appropriate plots based on the number and type of selected columns:
    - **1 Numerical**: Index Plot, Boxplot, Histogram.
    - **1 Categorical**: Barplot.
    - **2 Numerical**: Scatterplot.
    - **2 Categorical**: Heatmap.
    - **1 Numerical + 1 Categorical**: Boxplot and Density Plot (grouped by category).
    - **2 Numerical + 1 Categorical**: Scatterplot with points colored by the categorical variable.

### 4. Modeling
- **Model Selection**: Automatically selects **Linear Regression** for numerical targets and **AdaBoost Classifier** for categorical targets.
- **Configuration**:
    - Select **Target Variable**.
    - Select **Latent Variables** (features). If none selected, all other columns are used.
    - Adjust **Train/Test Split Ratio**.
- **Results**: Displays performance metrics (RMSE/MAE/R2 for Regression; Accuracy/Precision/F1 for Classification) on both Train and Test sets, along with a Variable Importance plot.

## Prerequisites

- **Git** installed.
- **Docker** installed (recommended for easiest setup).
- *(Optional)* **uv** if running locally without Docker.

## Running with Docker (Recommended)


1. **Clone the repository:**
   ```bash
   git clone https://github.com/berbomeister/csv-summarizer
   cd csv-summarizer/
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t csv-summarizer .
   ```

3. **Run the container:**
   ```bash
   docker run -p 8000:8000 csv-summarizer
   ```

4. **Access the App:**
   Open your web browser and navigate to `http://localhost:8000`.

## (Optionally) Running Locally with uv

If you prefer to run locally for development:

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the app:**
   ```bash
   uv run shiny run --reload src/app.py
   ```

