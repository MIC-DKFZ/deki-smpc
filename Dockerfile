FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get upgrade --yes \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --upgrade \
      pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
RUN groupadd --system deki && useradd --system --gid deki --home-dir /app deki
WORKDIR /app
COPY pyproject.toml setup.py README.md /app/
COPY deki_smpc /app/deki_smpc
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install --no-cache-dir . \
 && pip uninstall --yes pip wheel \
 && rm -rf /usr/local/lib/python3.12/ensurepip
COPY examples /app/examples
USER deki
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import deki_smpc"]
ENTRYPOINT ["python"]
