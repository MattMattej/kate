FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── Variables de entorno (sobreescribibles desde RunPod Environment) ──────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    # "sequential" usa ~12 GB VRAM (más lento); "model" usa ~20 GB (más rápido)
    CPU_OFFLOAD_TYPE=sequential \
    MAX_IMAGE_SIDE=768 \
    DEFAULT_STEPS=20 \
    LORA1_SCALE=0.85 \
    LORA2_SCALE=0.5 \
    # Evita fragmentación de VRAM en PyTorch
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Deshabilita telemetría innecesaria
    DIFFUSERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt handler.py ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python3", "-u", "handler.py"]
