#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SEINE

mkdir -p pretrained/stable-diffusion-v1-4 input/api results/api

if [[ ! -f pretrained/seine.pt ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
  fi

  hf download Vchitect/SEINE --local-dir pretrained
fi

if [[ ! -f pretrained/stable-diffusion-v1-4/model_index.json ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
  fi

  hf download CompVis/stable-diffusion-v1-4 --local-dir pretrained/stable-diffusion-v1-4
fi

if [[ ! -f pretrained/seine.pt ]]; then
  echo "Warning: pretrained/seine.pt is still missing after download; check HF access and model repo contents."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
