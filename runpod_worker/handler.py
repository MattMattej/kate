#!/usr/bin/env python3
"""
RunPod Worker — Pixel Kate NSFW (FLUX + dual LoRA)
Estrategia: LoRAs se cargan y fusionan UNA vez al arranque del worker.
Cada job solo ejecuta inferencia (rápido en RTX 4090 con model offload).
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

MODEL_ID = "camenduru/FLUX.1-dev-diffusers"
LORA_DIR = "/tmp/loras"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# model = más rápido en RTX 4090 24GB | sequential = más lento, suele hacer timeout
OFFLOAD_TYPE = os.environ.get("CPU_OFFLOAD_TYPE", "model").lower()
MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "512"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "16"))
FUSE_LORAS = os.environ.get("FUSE_LORAS", "1").strip() in ("1", "true", "yes")

ENV_LORA1_URL = os.environ.get("LORA1_URL", "").strip()
ENV_LORA2_URL = os.environ.get("LORA2_URL", "").strip()
DEFAULT_LORA1_SCALE = float(os.environ.get("LORA1_SCALE", "0.85"))
DEFAULT_LORA2_SCALE = float(os.environ.get("LORA2_SCALE", "0.5"))

_loras_ready = False
_lora_names_loaded: list[str] = []


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
    print(f"   📥 Descargando LoRA '{name}'...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   ✅ LoRA '{name}' en disco.")
    return local_path


def setup_loras_once(
    lora1_url: str,
    lora1_scale: float,
    lora2_url: str,
    lora2_scale: float,
) -> list[str]:
    """Carga + fusiona LoRAs una sola vez por vida del worker."""
    global _loras_ready, _lora_names_loaded

    if _loras_ready:
        print("   ♻️  LoRAs ya listos (sin recarga)")
        return _lora_names_loaded

    if not lora1_url and not lora2_url:
        enable_cpu_offload()
        _loras_ready = True
        return []

    t0 = time.time()
    print("   📦 Configurando LoRAs (una vez por worker)...")

    pipe.unload_lora_weights()
    pipe.to("cpu")
    free_gpu_memory()

    names: list[str] = []
    scales: list[float] = []

    if lora1_url:
        pipe.load_lora_weights(download_lora(lora1_url, "kate_identity"), adapter_name="kate")
        names.append("kate")
        scales.append(lora1_scale)
        print(f"   ✓ kate ({time.time() - t0:.0f}s)")

    if lora2_url:
        pipe.load_lora_weights(download_lora(lora2_url, "nsfw_style"), adapter_name="nsfw")
        names.append("nsfw")
        scales.append(lora2_scale)
        print(f"   ✓ nsfw ({time.time() - t0:.0f}s)")

    pipe.set_adapters(names, adapter_weights=scales)

    if FUSE_LORAS and names:
        print("   🔗 Fusionando LoRAs en GPU/CPU (~3-8 min, solo cold start)...")
        pipe.fuse_lora(adapter_names=names)
        print(f"   ✅ Fusion listo ({time.time() - t0:.0f}s)")
    else:
        print(f"   ✅ Adapters activos: {names} ({time.time() - t0:.0f}s)")

    free_gpu_memory()
    enable_cpu_offload()

    _loras_ready = True
    _lora_names_loaded = names
    return names


# ── Cold start ───────────────────────────────────────────────────────────────
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

os.makedirs(LORA_DIR, exist_ok=True)

print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"   Offload: {OFFLOAD_TYPE} | Max lado: {MAX_SIDE} | Steps default: {DEFAULT_STEPS}")

setup_loras_once(ENV_LORA1_URL, DEFAULT_LORA1_SCALE, ENV_LORA2_URL, DEFAULT_LORA2_SCALE)
print("✅ Worker listo para inferencia.")


def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "a beautiful woman, portrait")
    lora1_url = job_input.get("lora1_url", ENV_LORA1_URL) or ENV_LORA1_URL
    lora1_scale = float(job_input.get("lora1_scale", DEFAULT_LORA1_SCALE))
    lora2_url = job_input.get("lora2_url", ENV_LORA2_URL) or ENV_LORA2_URL
    lora2_scale = float(job_input.get("lora2_scale", DEFAULT_LORA2_SCALE))
    width = int(job_input.get("width", MAX_SIDE))
    height = int(job_input.get("height", MAX_SIDE))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    guidance = float(job_input.get("guidance", 3.5))
    seed = job_input.get("seed", None)

    width, height = clamp_resolution(width, height)

    print(f"\n🎨 Job: {prompt[:70]}...")
    print(f"   {width}x{height} | {steps} steps | offload={OFFLOAD_TYPE}")

    try:
        # Solo recarga si cambian URLs (raro); si no, inferencia directa
        if (lora1_url, lora2_url) != (ENV_LORA1_URL, ENV_LORA2_URL):
            global _loras_ready
            _loras_ready = False
            setup_loras_once(lora1_url, lora1_scale, lora2_url, lora2_scale)

        generator = torch.Generator("cuda").manual_seed(seed) if seed else None
        print(f"   ⚙️  Generando...")
        t0 = time.time()

        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            n = step_index + 1
            if n == 1 or n % 4 == 0 or n == steps:
                print(f"   📊 {n}/{steps} ({time.time() - t0:.0f}s)")
            return callback_kwargs

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=on_step_end,
            )

        elapsed = time.time() - t0
        print(f"   ✅ Imagen lista en {elapsed:.1f}s")

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
            "loras_used": _lora_names_loaded,
            "offload_type": OFFLOAD_TYPE,
        }

    except torch.cuda.OutOfMemoryError:
        free_gpu_memory()
        return {
            "error": "CUDA OOM — usa MAX_IMAGE_SIDE=512, DEFAULT_STEPS=16, FUSE_LORAS=1, GPU RTX 4090+"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        free_gpu_memory()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
