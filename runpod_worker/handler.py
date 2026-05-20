#!/usr/bin/env python3
"""
=============================================================================
RunPod Worker — Pixel Kate NSFW Generator
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
OFFLOAD_TYPE = os.environ.get("CPU_OFFLOAD_TYPE", "sequential").lower()
MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "768"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "24"))
# URLs por defecto (precarga al arranque — evita recargar/fusionar en cada job)
ENV_LORA1_URL = os.environ.get("LORA1_URL", "").strip()
ENV_LORA2_URL = os.environ.get("LORA2_URL", "").strip()
# ─────────────────────────────────────────────────────────────────────────────

# Estado global: LoRAs ya fusionados en el pipeline
_lora_cache_key: tuple | None = None


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def enable_cpu_offload():
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


def download_lora(url: str, name: str) -> str:
    local_path = os.path.join(LORA_DIR, f"{name}.safetensors")
    if os.path.exists(local_path):
        return local_path

    print(f"   📥 Descargando LoRA '{name}' desde: {url[:60]}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   ✅ LoRA '{name}' descargado.")
    return local_path


def apply_loras(lora1_url: str, lora1_scale: float, lora2_url: str, lora2_scale: float) -> list[str]:
    """
    Carga y fusiona LoRAs solo si cambió la combinación URL/escala.
    La fusión en CPU tarda varios minutos — no repetir en cada request.
    """
    global _lora_cache_key

    cache_key = (lora1_url, lora2_url, lora1_scale, lora2_scale)
    if cache_key == _lora_cache_key:
        print("   ♻️  LoRAs ya cargados y fusionados (caché)")
        return [n for n, u in [("kate", lora1_url), ("nsfw", lora2_url)] if u]

    t0 = time.time()
    print("   📦 Cargando y fusionando LoRAs (solo cuando cambian URLs/escalas)...")

    try:
        pipe.unfuse_lora()
    except Exception:
        pass

    pipe.remove_all_hooks()
    pipe.unload_lora_weights()
    pipe.to("cpu")
    free_gpu_memory()

    lora_names: list[str] = []
    lora_scales: list[float] = []

    if lora1_url:
        path = download_lora(lora1_url, "kate_identity")
        pipe.load_lora_weights(path, adapter_name="kate")
        lora_names.append("kate")
        lora_scales.append(lora1_scale)
        print(f"   ✓ LoRA kate cargado ({time.time() - t0:.1f}s)")

    if lora2_url:
        path = download_lora(lora2_url, "nsfw_style")
        pipe.load_lora_weights(path, adapter_name="nsfw")
        lora_names.append("nsfw")
        lora_scales.append(lora2_scale)
        print(f"   ✓ LoRA nsfw cargado ({time.time() - t0:.1f}s)")

    if lora_names:
        print("   🔗 Fusionando LoRAs en transformer (puede tardar 2-5 min)...")
        pipe.set_adapters(lora_names, adapter_weights=lora_scales)
        pipe.fuse_lora(adapter_names=lora_names)
        print(f"   ✅ LoRAs fusionados: {lora_names} escalas {lora_scales} ({time.time() - t0:.1f}s total)")

    free_gpu_memory()
    print("   ⚙️  Activando CPU offload...")
    enable_cpu_offload()
    _lora_cache_key = cache_key
    return lora_names


# ── Arranque del worker (cold start) ─────────────────────────────────────────
print("🔄 Cargando FLUX.1-dev...")
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
    print("   ⚙️  Sequential CPU offload (ahorro VRAM)")
else:
    print("   ⚙️  Model CPU offload")
enable_cpu_offload()
print("✅ Modelo base cargado.")

os.makedirs(LORA_DIR, exist_ok=True)

if ENV_LORA1_URL or ENV_LORA2_URL:
    print("🔄 Precargando LoRAs desde variables de entorno...")
    apply_loras(ENV_LORA1_URL, 0.85, ENV_LORA2_URL, 0.5)
    print("✅ LoRAs precargados — el primer job será mucho más rápido.")


def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "a beautiful woman, portrait")
    lora1_url = job_input.get("lora1_url", ENV_LORA1_URL)
    lora1_scale = float(job_input.get("lora1_scale", 0.85))
    lora2_url = job_input.get("lora2_url", ENV_LORA2_URL)
    lora2_scale = float(job_input.get("lora2_scale", 0.5))
    width = int(job_input.get("width", MAX_SIDE))
    height = int(job_input.get("height", MAX_SIDE))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    guidance = float(job_input.get("guidance", 3.5))
    seed = job_input.get("seed", None)

    width, height = clamp_resolution(width, height)

    print(f"\n🎨 Nuevo job recibido:")
    print(f"   Prompt: {prompt[:80]}")
    print(f"   Resolución: {width}x{height} | Steps: {steps} | Offload: {OFFLOAD_TYPE}")

    try:
        lora_names = apply_loras(lora1_url, lora1_scale, lora2_url, lora2_scale)

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


runpod.serverless.start({"handler": handler})
