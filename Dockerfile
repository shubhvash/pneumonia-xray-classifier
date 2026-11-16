# Serving image: FastAPI + ONNX Runtime only (no PyTorch) -> small & fast to start
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PORT=9696

WORKDIR /app

# 1. install only the runtime dependencies from the lock file (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project && rm -rf /root/.cache

# 2. application code and model artefacts
COPY predict.py ./
COPY models/ ./models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 9696
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# PORT is overridable (Render injects its own)
CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]
