WORKDIR /app

COPY src/ ./src/
COPY data/ ./data/
COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync

EXPOSE 8000

CMD [ "uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]