FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
    FACILITY_ENV=production FACILITY_RATE_DB=/tmp/facility-rate.sqlite3 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements-runtime-lock.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements-runtime-lock.txt \
    && pip uninstall -y setuptools pip \
    && useradd --uid 10001 --create-home facility
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 app ./app
COPY --chown=10001:10001 data ./data
COPY --chown=10001:10001 models ./models
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--timeout-keep-alive", "5", "--limit-concurrency", "32", "--backlog", "64", "--timeout-graceful-shutdown", "15"]
