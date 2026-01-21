# Use a specific version of the prebuilt uv image
FROM ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY uv.lock ./

# Sync dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run command
CMD ["uv", "run", "shiny", "run", "--host", "0.0.0.0", "--port", "8000", "src/app.py"]
