# E2M on an NVIDIA GPU (Docker Desktop with WSL2 on Windows, or Linux with the NVIDIA container toolkit).
# Build:  docker build -t e2m --build-arg E2M_COMMIT=$(git rev-parse HEAD) .
# Run:    docker run --gpus all --rm -v <data folder>:/data -v <output folder>:/runs e2m \
#             cv --cancer LUAD --data-dir /data --output /runs/luad_cv
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PATH=/opt/venv/bin:$PATH

# A virtual environment with a current pip; Ubuntu's system pip cannot build from pyproject.toml.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv \
    && python -m pip install pip==26.1.2

WORKDIR /opt/e2m
COPY requirements-lock.txt .
RUN python -m pip install -r requirements-lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-deps . && python -m pip freeze > /opt/e2m/installed.txt

ARG E2M_COMMIT=unknown
ENV E2M_COMMIT=${E2M_COMMIT}

WORKDIR /work
ENTRYPOINT ["e2m"]
CMD ["--help"]
