# Railway Buddy — one image, two roles (API or UI), selected by the command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    RB_STATE_DIR=/app/runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 rb && mkdir -p /app/runtime && chown -R rb /app
USER rb

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://127.0.0.1:8000/api/health || curl -fs http://127.0.0.1:8501/_stcore/health || exit 1

# Default: everything in one container (same as Streamlit Cloud). docker-compose runs API and UI separately.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
