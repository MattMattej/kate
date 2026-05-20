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

import os
import io
import time
import base64
import runpod
import torch
import requests
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download

# ── Configuración ────────────────────────────────────────────────────────────
MODEL_ID  = "camenduru/FLUX.1-dev-diffusers"
LORA_DIR  = "/tmp/loras"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")  # Ya no es estrictamente necesario gracias al mirror libre
# ─────────────────────────────────────────────────────────────────────────────

# Cargamos el pipeline UNA sola vez al arrancar el worker (no en cada request)
print("🔄 Cargando FLUX.1-dev en GPU...")
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN if HF_TOKEN else None,
)
pipe.enable_model_cpu_offload()  # Gestión inteligente de VRAM
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

    # Parámetros del request
    prompt      = job_input.get("prompt", "a beautiful woman, portrait")
    lora1_url   = job_input.get("lora1_url", "")
    lora1_scale = float(job_input.get("lora1_scale", 0.85))
    lora2_url   = job_input.get("lora2_url", "")
    lora2_scale = float(job_input.get("lora2_scale", 0.5))
    width       = int(job_input.get("width", 1024))
    height      = int(job_input.get("height", 1024))
    steps       = int(job_input.get("steps", 28))
    guidance    = float(job_input.get("guidance", 3.5))
    seed        = job_input.get("seed", None)

    print(f"\n🎨 Nuevo job recibido:")
    print(f"   Prompt: {prompt[:80]}")
    print(f"   LoRA1 scale: {lora1_scale} | LoRA2 scale: {lora2_scale}")

    try:
        # ── Cargar LoRAs dinámicamente ────────────────────────────────────────
        # Primero descargar los LoRAs si se pasaron URLs
        lora1_path = None
        lora2_path = None

        if lora1_url:
            lora1_path = download_lora(lora1_url, "kate_identity")
        if lora2_url:
            lora2_path = download_lora(lora2_url, "nsfw_style")

        # ⚠️ CRÍTICO: Para evitar errores de CUDA (device mismatch / assertion),
        # removemos temporalmente todos los hooks de CPU offload. Esto devuelve el
        # pipeline a su estado limpio para que PEFT pueda modificar los pesos sin interferencia.
        print("   📦 Desactivando hooks de CPU offload temporalmente...")
        pipe.remove_all_hooks()

        # Aplicar LoRAs al pipeline usando diffusers
        # Necesitamos unload primero para limpiar LoRAs anteriores
        pipe.unload_lora_weights()

        lora_paths  = []
        lora_names  = []
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
            print(f"   ✅ LoRAs activos: {lora_names} con escalas {lora_scales}")

        # ⚠️ Re-activamos CPU offload para liberar VRAM antes de correr la inferencia
        print("   ⚙️  Re-activando CPU offload para inferencia...")
        pipe.enable_model_cpu_offload()

        # ── Generar imagen ────────────────────────────────────────────────────
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

        # ── Convertir a base64 para devolver en el response ───────────────────
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
            "loras_used": lora_names
        }

    except Exception as e:
        print(f"❌ Error en el worker: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Punto de entrada de RunPod
runpod.serverless.start({"handler": handler})
