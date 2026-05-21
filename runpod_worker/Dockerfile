# Pixel Kate — RunPod Worker
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    CPU_OFFLOAD_TYPE=model \
    MAX_IMAGE_SIDE=512 \
    DEFAULT_STEPS=16 \
    FUSE_LORAS=1 \
    LORA1_SCALE=0.85 \
    LORA2_SCALE=0.5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY handler.py .

EXPOSE 8000
CMD ["python3", "-u", "handler.py"]
