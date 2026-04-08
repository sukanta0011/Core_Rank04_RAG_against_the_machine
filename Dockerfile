# 1. Use a high-performance Python image with 'uv' pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files and enable bytecode compilation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# 4. Copy dependency files first (to use Docker's cache system)
COPY pyproject.toml uv.lock ./

# 5. Install dependencies without the project itself
# This makes subsequent builds much faster
RUN uv sync --frozen --no-install-project

# 6. Copy your source code and data folders
COPY src/ ./src/
COPY data/ ./data/

# 7. Final sync to install the project package
RUN uv sync --frozen

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. The command to run your API
# We use 0.0.0.0 so the container can be accessed from outside
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]