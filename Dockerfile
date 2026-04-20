# OpsTwin Recovery Arena server image.
# Starts the FastAPI app on port 8000.
FROM python:3.12-slim

WORKDIR /app

# Install only the runtime deps. Training/eval happen outside the container.
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.2" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0"

# Copy repo. Order matters: code that changes most last so docker caches well.
COPY models.py /app/models.py
COPY server /app/server

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
