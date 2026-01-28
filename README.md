# CSV Summarizer App

This project contains all the necessary code to create and run a Shiny for Python application that allows users to upload CSV files, view statistics, create visualizations, and train simple machine learning models.

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

