from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from faicons import icon_svg

# Adjust imports for running as a script vs package
try:
    from .plotting import (
        plot_numerical_index, plot_numerical_boxplot, plot_numerical_histogram,
        plot_categorical_bar, plot_scatterplot, plot_grouped_boxplot,
        plot_grouped_density, plot_categorical_heatmap, plot_correlation_matrix,
        plot_variable_importance
    )
    from .stats import get_dataset_stats, get_column_stats, get_column_summary_df
    from .modeling import train_linear_model, train_adaboost_model
    from .processing import impute_data, get_missing_columns
except ImportError:
    # Fallback for running directly if not installed as package
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.plotting import (
        plot_numerical_index, plot_numerical_boxplot, plot_numerical_histogram,
        plot_categorical_bar, plot_scatterplot, plot_grouped_boxplot,
        plot_grouped_density, plot_categorical_heatmap, plot_correlation_matrix,
        plot_variable_importance
    )
    from src.stats import get_dataset_stats, get_column_stats, get_column_summary_df
    from src.modeling import train_linear_model, train_adaboost_model
    from src.processing import impute_data, get_missing_columns

# Reproducibility
np.random.seed(42)

app_ui = ui.page_navbar(
    ui.nav_panel("Welcome",
        ui.card(
            ui.card_header("Welcome to the CSV Summarizer App"),
            ui.markdown(
                """
                This application allows you to explore, visualize, and analyze your CSV datasets interactively.
                
                ### Features & Functionalities:
                
                **1. Data & Stats**
                - **Upload**: Load your own CSV file (default is the Iris dataset).
                - **Overview**: View raw data and dataset-level statistics.
                - **Variable Types**: Manage which columns are Numerical vs. Categorical.
                - **Statistics**: See detailed summary statistics for all columns.
                - **Missing Data**: Handle missing values with various strategies (Mean, Median, Mode, Drop Rows, etc.).
                
                **2. Visualizations**
                - **Correlation**: View a correlation matrix of the dataset.
                - **Dynamic Plots**: Select up to 3 columns to generate specific visualizations:
                    - *1 Column*: Histogram, Boxplot (Numerical) or Barplot (Categorical).
                    - *2 Columns*: Scatterplot (Num/Num), Heatmap (Cat/Cat), Boxplot/Density (Num/Cat).
                    - *3 Columns*: Scatterplot with Color (2 Num + 1 Cat).
                
                **3. Modeling**
                - **Train Models**: Build a Linear Regression (Numerical Target) or AdaBoost Classifier (Categorical Target).
                - **Configuration**: Select Target and Latent variables, and adjust the Train/Test split ratio.
                - **Results**: View performance metrics and variable importance.
                """
            )
        )
    ),
    ui.nav_panel("Data & Stats",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_file("file_upload", "Upload CSV", accept=[".csv"]),
            ),
            ui.card(
                ui.card_header("Dataset Overview"),
                ui.output_data_frame("data_preview"),
                ui.output_text_verbatim("dataset_stats_text"),
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("All Column Statistics"),
                    ui.output_data_frame("all_columns_stats")
                ),
                ui.card(
                    ui.card_header("Variable Types Management"),
                    ui.help_text("Check variables that should be treated as CATEGORICAL. Unchecked variables will be converted to NUMERICAL."),
                    ui.input_checkbox_group("cat_cols_selection", "Select Categorical Variables", choices=[]),
                    ui.input_action_button("apply_types_btn", "Apply Type Changes", class_="btn-primary")
                ),
                width=1/2
            ),
            ui.card(
                ui.card_header("Missing Data Handling"),
                ui.layout_column_wrap(
                     ui.input_select("impute_col", "Select Column (with missing values)", choices=[]),
                     ui.input_select("impute_strategy", "Imputation Strategy", choices=[]),
                     ui.input_action_button("apply_impute_btn", "Apply Imputation", class_="btn-warning"),
                     width=1/3
                )
            )
        )
    ),
    ui.nav_panel("Visualizations",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize("viz_cols", "Select Columns to Visualize", choices=[], multiple=True),
                ui.help_text("Select columns to see specific plots (Max 3).")
            ),
            ui.card(
                ui.card_header("Dataset Correlation"),
                ui.output_plot("viz_corr_matrix")
            ),
            ui.card(
                ui.card_header("Column Specific Plots"),
                ui.output_ui("viz_dynamic_container")
            )
        )
    ),
    ui.nav_panel("Modeling",
         ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("model_target", "Target Variable", choices=[]),
                ui.input_selectize("model_features", "Latent Variables", choices=[], multiple=True),
                ui.input_slider("train_split", "Train Split Ratio", 0.0, 0.99, 0.8, step=0.05),
                ui.input_action_button("train_btn", "Train Model", class_="btn-primary")
            ),
            ui.card(
                ui.card_header("Model Metrics"),
                ui.output_text_verbatim("model_metrics")
            ),
            ui.card(
                ui.card_header("Variable Importance"),
                ui.output_plot("model_importance_plot")
            )
        )
    ),
    title="CSV Summarizer"
)

def server(input, output, session):
    # Reactive value to hold the dataframe
    val = reactive.Value(sns.load_dataset("iris"))
    
    @reactive.Effect
    @reactive.event(input.file_upload)
    def load_file():
        file_infos = input.file_upload()
        if not file_infos:
            return
        df = pd.read_csv(file_infos[0]["datapath"])
        val.set(df)
        
    @reactive.Effect
    def update_column_choices():
        df = val.get()
        cols = list(df.columns)
        
        # Update viz and model choices
        ui.update_selectize("viz_cols", choices=cols)
        ui.update_select("model_target", choices=cols)
        ui.update_selectize("model_features", choices=cols)
        
        # Update checkbox group for types
        # Identify currently categorical columns
        current_cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        
        # Create choices with some info
        # But for checkbox group, simple choices are better for ID stability
        ui.update_checkbox_group("cat_cols_selection", choices=cols, selected=current_cats)
        
        # Update missing data column choices
        missing_cols = get_missing_columns(df)
        ui.update_select("impute_col", choices=missing_cols)

    @reactive.Effect
    def update_impute_strategies():
        col = input.impute_col()
        df = val.get()
        if not col or col not in df.columns:
            return
            
        is_num = pd.api.types.is_numeric_dtype(df[col])
        if is_num:
            strategies = {
                "Drop Rows": "drop_rows",
                "Fill with Mean": "mean",
                "Fill with Median": "median",
                "Fill with 0": "zero",
                "Fill with Mode": "mode"
            }
        else:
            strategies = {
                "Drop Rows": "drop_rows",
                "Fill with Mode": "mode",
                "Fill with 'Missing'": "missing_label"
            }
        ui.update_select("impute_strategy", choices=strategies)
        
    @reactive.Effect
    @reactive.event(input.apply_impute_btn)
    def apply_imputation():
        col = input.impute_col()
        strategy = input.impute_strategy()
        
        if not col or not strategy:
            return
            
        try:
            df = val.get()
            clean_df = impute_data(df, col, strategy)
            val.set(clean_df)
            ui.notification_show(f"Applied imputation '{strategy}' on '{col}'", type="message")
            
            # Force update of choices in case missing columns changed (e.g. removed from list)
            # This is partly handled by the Effect on val.set -> update_column_choices
        except Exception as e:
            ui.notification_show(f"Error imputing data: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.apply_types_btn)
    def bulk_update_types():
        selected_cats = input.cat_cols_selection()
        df = val.get().copy()
        cols = df.columns
        
        changes = []
        errors = []
        
        for col in cols:
            should_be_cat = col in selected_cats
            is_currently_num = pd.api.types.is_numeric_dtype(df[col])
            
            try:
                if should_be_cat and is_currently_num:
                    # Convert to categorical (string)
                    df[col] = df[col].astype(str)
                    changes.append(f"{col} -> Categorical")
                elif not should_be_cat and not is_currently_num:
                    # Convert to numeric
                    # pd.to_numeric can fail or produce NaN
                    # We use coerce to turn failures into NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    changes.append(f"{col} -> Numerical")
            except Exception as e:
                errors.append(f"{col}: {str(e)}")
        
        if changes or errors:
            val.set(df)
            msg = ""
            if changes:
                msg += "Updated: " + ", ".join(changes)
            if errors:
                msg += " | Errors: " + ", ".join(errors)
            
            type_msg = "error" if errors else "message"
            ui.notification_show(msg, type=type_msg)
            
    # --- Data & Stats ---
    
    @render.data_frame
    def data_preview():
        return render.DataGrid(val.get().head(10))
        
    @render.text
    def dataset_stats_text():
        stats = get_dataset_stats(val.get())
        return "\n".join([f"{k}: {v}" for k, v in stats.items()])
        
    @render.data_frame
    def all_columns_stats():
        return render.DataGrid(get_column_summary_df(val.get()))
        
    # --- Visualizations ---
    
    @render.plot
    def viz_corr_matrix():
        return plot_correlation_matrix(val.get())
    
    @render.ui
    def viz_dynamic_container():
        # Based on selected columns, return layout of plots
        cols = input.viz_cols()
        df = val.get()
        
        if not cols:
            return ui.p("Please select columns.")
            
        # Analyze types
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        
        plots = []
        
        # Logic from requirements
        if len(cols) == 1:
            c = cols[0]
            if c in num_cols:
                # 3 plots: Index, Box, Hist
                plots.append(ui.output_plot("plot_1_index"))
                plots.append(ui.output_plot("plot_1_box"))
                plots.append(ui.output_plot("plot_1_hist"))
            else:
                # 1 plot: Bar
                plots.append(ui.output_plot("plot_1_bar"))
                
        elif len(cols) == 2:
            if len(num_cols) == 2:
                # Scatter
                plots.append(ui.output_plot("plot_2_scatter"))
            elif len(cat_cols) == 2:
                # Heatmap
                plots.append(ui.output_plot("plot_2_heatmap"))
            else:
                # Num + Cat
                plots.append(ui.output_plot("plot_2_box"))
                plots.append(ui.output_plot("plot_2_density"))
                
        elif len(cols) == 3:
            if len(num_cols) == 2 and len(cat_cols) == 1:
                # Scatter with hue
                plots.append(ui.output_plot("plot_3_scatter"))
            else:
                return ui.p("This combination of 3 columns is not supported (need 2 Num + 1 Cat).")
        
        else:
            return ui.p("Please select 1, 2, or 3 columns (supported combinations only).")
            
        return ui.layout_column_wrap(*plots, width=1/len(plots) if len(plots) > 0 else 1)

    # We need to define the render functions for these dynamic IDs
    # Since Shiny requires defined render functions, we can define them all and they will simply not be shown if not in UI
    # Or better: use a single render.ui that generates the plots directly? 
    # Shiny for Python handles 'render.plot' returning a figure. 
    # But creating dynamic output IDs is tricky.
    # Actually, the best way for dynamic number of plots in Shiny Python is usually rendering a single figure with subplots 
    # OR using @render.ui to return `ui.output_plot`s and having defined `@render.plot` functions for fixed names.
    
    # I used fixed names above ("plot_1_index", etc). I just need to define them.
    
    @render.plot
    def plot_1_index():
        cols = input.viz_cols()
        if len(cols) == 1 and cols[0] in val.get().select_dtypes(include=np.number).columns:
            return plot_numerical_index(val.get(), cols[0])
            
    @render.plot
    def plot_1_box():
        cols = input.viz_cols()
        if len(cols) == 1 and cols[0] in val.get().select_dtypes(include=np.number).columns:
            return plot_numerical_boxplot(val.get(), cols[0])

    @render.plot
    def plot_1_hist():
        cols = input.viz_cols()
        if len(cols) == 1 and cols[0] in val.get().select_dtypes(include=np.number).columns:
            return plot_numerical_histogram(val.get(), cols[0])
            
    @render.plot
    def plot_1_bar():
        cols = input.viz_cols()
        if len(cols) == 1 and cols[0] not in val.get().select_dtypes(include=np.number).columns:
            return plot_categorical_bar(val.get(), cols[0])
            
    @render.plot
    def plot_2_scatter():
        cols = input.viz_cols()
        if len(cols) == 2:
            nums = [c for c in cols if pd.api.types.is_numeric_dtype(val.get()[c])]
            if len(nums) == 2:
                return plot_scatterplot(val.get(), nums[0], nums[1])

    @render.plot
    def plot_2_heatmap():
        cols = input.viz_cols()
        if len(cols) == 2:
            cats = [c for c in cols if not pd.api.types.is_numeric_dtype(val.get()[c])]
            if len(cats) == 2:
                return plot_categorical_heatmap(val.get(), cats[0], cats[1])
                
    @render.plot
    def plot_2_box():
        cols = input.viz_cols()
        if len(cols) == 2:
            df = val.get()
            nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) == 1 and len(cats) == 1:
                return plot_grouped_boxplot(df, nums[0], cats[0])

    @render.plot
    def plot_2_density():
        cols = input.viz_cols()
        if len(cols) == 2:
            df = val.get()
            nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) == 1 and len(cats) == 1:
                return plot_grouped_density(df, nums[0], cats[0])
                
    @render.plot
    def plot_3_scatter():
        cols = input.viz_cols()
        if len(cols) == 3:
            df = val.get()
            nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) == 2 and len(cats) == 1:
                return plot_scatterplot(df, nums[0], nums[1], hue=cats[0])

    # --- Modeling ---
    
    metrics_val = reactive.Value("")
    importance_val = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.train_btn)
    def train_model():
        df = val.get()
        target = input.model_target()
        features = list(input.model_features())
        split = input.train_split()
        
        if not target:
            metrics_val.set("Please select a target variable.")
            importance_val.set(None)
            return

        if not features:
            features = [c for c in df.columns if c != target]
            
        test_size = 1.0 - split
        if test_size <= 0 or test_size >= 1:
             metrics_val.set("Invalid split ratio.")
             return
        
        try:
            if pd.api.types.is_numeric_dtype(df[target]):
                # Linear
                metrics, importance = train_linear_model(df, target, features, test_size)
            else:
                # AdaBoost
                metrics, importance = train_adaboost_model(df, target, features, test_size)
                
            metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            metrics_val.set(metrics_str)
            importance_val.set(importance)
            
        except Exception as e:
            metrics_val.set(f"Error training model: {str(e)}")
            importance_val.set(None)
            
    @render.text
    def model_metrics():
        return metrics_val.get()
        
    @render.plot
    def model_importance_plot():
        imp = importance_val.get()
        if imp is not None:
            return plot_variable_importance(imp)

app = App(app_ui, server)
