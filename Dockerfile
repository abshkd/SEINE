FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/SEINE

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python --version \
    && pip --version \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt /tmp/requirement.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio \
    && python -m pip install --no-cache-dir "xformers>=0.0.35" \
    && python -m pip install --no-cache-dir "huggingface_hub[cli]" fastapi "uvicorn[standard]" python-multipart \
    && grep -vE "^(--extra-index-url|torch|torchvision|torchaudio|xformers)" /tmp/requirement.txt > /tmp/requirement.no_torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirement.no_torch.txt

COPY . /workspace/SEINE

RUN chmod +x /workspace/SEINE/docker/entrypoint.sh \
    && mkdir -p /workspace/SEINE/input/api /workspace/SEINE/results/api /workspace/SEINE/pretrained

EXPOSE 8000

CMD ["/workspace/SEINE/docker/entrypoint.sh"]
