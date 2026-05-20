# ============================================================
# Pixel Kate — RunPod Worker Dockerfile
# ============================================================
# Usamos la imagen oficial de PyTorch con CUDA 12.1 pre-instalado
# Esto reduce el tiempo de compilación en GitHub Actions de 6 minutos a solo 30 segundos!
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Variables de build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar y instalar dependencias Python primero (mejor cache de Docker)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copiar el handler
COPY handler.py .

# Pre-descargar el modelo FLUX.1-dev durante el build para arranques rápidos
# Necesita HF_TOKEN si el modelo es privado/gated
# ARG HF_TOKEN
# RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', token='${HF_TOKEN}')"
# NOTA: Si comentas esto, el modelo se descarga la primera vez que arranca el worker
# (tarda 5-10 min el primer arranque). Con un Network Volume montado en /workspace
# se cachea automáticamente entre sesiones.

# Puerto de salud del worker (RunPod lo usa internamente)
EXPOSE 8000

# Comando de inicio
CMD ["python3", "-u", "handler.py"]
