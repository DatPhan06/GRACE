FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Prevent venv from being shadowed by bind mount
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Install system dependencies
# Use cache mount to speed up apt-get
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies definitions
COPY pyproject.toml uv.lock ./

# Install dependencies using uv with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the project files
COPY . .

# Install the project with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Expose port (default 8000, but compose might map it)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
