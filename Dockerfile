# Multi-stage build: compile the Vite frontend before packaging the FastAPI service.

# --- Frontend build stage ----------------------------------------------------
FROM node:18-alpine AS frontend-build
WORKDIR /app

# Install dependencies before copying the full source to leverage Docker cache.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy the remaining frontend sources and build the production bundle.
COPY frontend/ ./
ARG VITE_API_BASE_URL=/
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# --- Backend runtime stage ---------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies and the evaluation agent package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Copy the built frontend artifacts from the previous stage.
COPY --from=frontend-build /app/dist ./frontend/dist

# Ensure directories expected by the application exist at runtime.
RUN mkdir -p runs

EXPOSE 8000

CMD ["sh", "-c", "python -m eval_agent.cli serve --host 0.0.0.0 --port ${PORT:-8000}"]
