#!/usr/bin/env python3
"""
=============================================================================
RunPod Worker — Pixel Kate NSFW Generator
=============================================================================
Este archivo es el "cerebro" que corre DENTRO del servidor de RunPod.
Recibe un request JSON con el prompt y las URLs de los LoRAs,
genera la imagen con FLUX Dev y devuelve la imagen en base64.
=============================================================================
"""

import gc
import os
import io
import time
import base64
import runpod
import torch
import requests
from diffusers import FluxPipeline

# ── Configuración ────────────────────────────────────────────────────────────
MODEL_ID = "camenduru/FLUX.1-dev-diffusers"
LORA_DIR = "/tmp/loras"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# sequential = capas una a una (menos VRAM). model = módulo entero en GPU (más rápido, OOM con dual LoRA)
OFFLOAD_TYPE = os.environ.get("CPU_OFFLOAD_TYPE", "sequential").lower()
# Resolución máxima si el cliente pide más (protege GPUs de 24GB)
MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
# ─────────────────────────────────────────────────────────────────────────────


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def enable_cpu_offload():
    """Aplica CPU offload limpio (sin hooks duplicados)."""
    pipe.remove_all_hooks()
    if OFFLOAD_TYPE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()


def clamp_resolution(width: int, height: int) -> tuple[int, int]:
    side = max(width, height)
    if side <= MAX_SIDE:
        return width, height
    scale = MAX_SIDE / side
    return int(width * scale), int(height * scale)


# Cargamos el pipeline UNA sola vez al arrancar el worker (no en cada request)
print("🔄 Cargando FLUX.1-dev en GPU...")
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN if HF_TOKEN else None,
)
pipe.enable_attention_slicing("max")
if getattr(pipe, "vae", None) is not None:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

if OFFLOAD_TYPE == "sequential":
    print("   ⚙️  Activando sequential CPU offload (modo ahorro de VRAM)...")
else:
    print("   ⚙️  Activando model CPU offload (requiere GPU ≥40GB con dual LoRA)...")
enable_cpu_offload()
print("✅ Modelo base cargado.")

os.makedirs(LORA_DIR, exist_ok=True)


def download_lora(url: str, name: str) -> str:
    """Descarga un LoRA desde una URL pública y devuelve la ruta local."""
    local_path = os.path.join(LORA_DIR, f"{name}.safetensors")
    if os.path.exists(local_path):
        print(f"   ♻️  LoRA '{name}' ya está en caché.")
        return local_path

    print(f"   📥 Descargando LoRA '{name}' desde: {url[:60]}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   ✅ LoRA '{name}' descargado.")
    return local_path


def handler(job):
    """
    Función principal del worker. RunPod llama esto por cada request.

    Input esperado (job["input"]):
        prompt       (str)   : Descripción de la imagen
        lora1_url    (str)   : URL directa al .safetensors del LoRA de Kate
        lora1_scale  (float) : Intensidad del LoRA de Kate (default: 0.85)
        lora2_url    (str)   : URL directa al .safetensors del LoRA NSFW (opcional)
        lora2_scale  (float) : Intensidad del LoRA NSFW (default: 0.5)
        width        (int)   : Ancho de la imagen (default: 1024)
        height       (int)   : Alto de la imagen (default: 1024)
        steps        (int)   : Pasos de inferencia (default: 28)
        guidance     (float) : Guidance scale (default: 3.5)
        seed         (int)   : Semilla aleatoria (opcional)
    """
    job_input = job["input"]

    prompt = job_input.get("prompt", "a beautiful woman, portrait")
    lora1_url = job_input.get("lora1_url", "")
    lora1_scale = float(job_input.get("lora1_scale", 0.85))
    lora2_url = job_input.get("lora2_url", "")
    lora2_scale = float(job_input.get("lora2_scale", 0.5))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("steps", 28))
    guidance = float(job_input.get("guidance", 3.5))
    seed = job_input.get("seed", None)

    width, height = clamp_resolution(width, height)

    print(f"\n🎨 Nuevo job recibido:")
    print(f"   Prompt: {prompt[:80]}")
    print(f"   LoRA1 scale: {lora1_scale} | LoRA2 scale: {lora2_scale}")
    print(f"   Resolución: {width}x{height} | Offload: {OFFLOAD_TYPE}")

    try:
        lora1_path = download_lora(lora1_url, "kate_identity") if lora1_url else None
        lora2_path = download_lora(lora2_url, "nsfw_style") if lora2_url else None

        # Liberar VRAM: quitar hooks, deshacer fusión previa y cargar LoRAs en CPU
        print("   📦 Preparando LoRAs (CPU, sin hooks)...")
        try:
            pipe.unfuse_lora()
        except Exception:
            pass

        pipe.remove_all_hooks()
        pipe.unload_lora_weights()
        pipe.to("cpu")
        free_gpu_memory()

        lora_names = []
        lora_scales = []

        if lora1_path:
            pipe.load_lora_weights(lora1_path, adapter_name="kate")
            lora_names.append("kate")
            lora_scales.append(lora1_scale)

        if lora2_path:
            pipe.load_lora_weights(lora2_path, adapter_name="nsfw")
            lora_names.append("nsfw")
            lora_scales.append(lora2_scale)

        if lora_names:
            pipe.set_adapters(lora_names, adapter_weights=lora_scales)
            # Fusionar adapters reduce VRAM en inferencia (crítico con dual LoRA)
            pipe.fuse_lora(adapter_names=lora_names)
            print(f"   ✅ LoRAs fusionados: {lora_names} escalas {lora_scales}")

        free_gpu_memory()
        print("   ⚙️  Re-activando CPU offload para inferencia...")
        enable_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(seed) if seed else None

        print(f"   ⚙️  Generando {width}x{height} con {steps} pasos...")
        t0 = time.time()

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        elapsed = time.time() - t0
        print(f"   ✅ Imagen generada en {elapsed:.1f}s")

        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image_base64": img_b64,
            "format": "jpeg",
            "width": width,
            "height": height,
            "generation_time_seconds": round(elapsed, 2),
            "loras_used": lora_names,
            "offload_type": OFFLOAD_TYPE,
        }

    except Exception as e:
        print(f"❌ Error en el worker: {e}")
        import traceback

        traceback.print_exc()
        free_gpu_memory()
        return {"error": str(e)}

    finally:
        # Dejar el worker listo para el siguiente job sin fugas de VRAM
        try:
            pipe.unfuse_lora()
        except Exception:
            pass
        free_gpu_memory()


# Punto de entrada de RunPod
runpod.serverless.start({"handler": handler})
