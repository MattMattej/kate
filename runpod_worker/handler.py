#!/usr/bin/env python3
"""
Pixel Kate — RunPod Worker v7  (PRODUCCIÓN - ROBUSTA)
=====================================================
Historial de bugs resueltos:
  v1-v3 : TypeError — se pasaba lista a load_lora_weights en vez de str
  v4     : fuse_lora() sobre modelo con sequential_cpu_offload activo → hang infinito
  v5     : remove_all_hooks() mal ubicado → hook conflict en diffusers 0.30.x
  v6     : orden correcto: carga LoRAs → set_adapters → offload (sin fuse, sin hooks extra)
  v7 ✅  : validación robusta, mejor logging GPU, callbacks mejorados, manejo completo de errores

Reglas de oro (NO CAMBIAR):
  1. NUNCA llamar fuse_lora() con offload activo (causa hang infinito)
  2. NUNCA remove_all_hooks() antes de enable_*_cpu_offload() (destruye hooks internos)
  3. Cada load_lora_weights() recibe UN str path NUNCA lista
  4. Server RunPod arranca PRIMERO, modelo carga lazy en el primer job
  5. Validar GPU state y tensor shape ANTES de inferencia
"""

import gc
import os
import io
import sys
import time
import base64
import traceback
import runpod
import torch
import requests
from diffusers import FluxPipeline

# ── Identidad del build (cambia en cada deploy para verificar en logs) ─────────
BUILD_ID = "pixel-kate-v7-production"

# ── Configuración desde variables de entorno RunPod ───────────────────────────
MODEL_ID      = "camenduru/FLUX.1-dev-diffusers"
LORA_DIR      = "/tmp/loras"
HF_TOKEN      = os.environ.get("HF_TOKEN", "").strip()

# CPU_OFFLOAD_TYPE: "sequential" (menor VRAM) o "model" (más rápido, ~20 GB VRAM)
OFFLOAD_TYPE  = os.environ.get("CPU_OFFLOAD_TYPE", "sequential").lower()
MAX_SIDE      = int(os.environ.get("MAX_IMAGE_SIDE", os.environ.get("MAX_SIDE", "768")))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "20"))

# URLs de LoRAs — pueden sobreescribirse en RunPod Environment
LORA1_URL = os.environ.get(
    "LORA1_URL",
    "https://v3b.fal.media/files/b/0a9aca2a/hToqwVnqjQOWYiHgKNx9O_pytorch_lora_weights.safetensors",
).strip()
LORA2_URL = os.environ.get(
    "LORA2_URL",
    "https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors",
).strip()
LORA1_SCALE = float(os.environ.get("LORA1_SCALE", "0.85"))
LORA2_SCALE = float(os.environ.get("LORA2_SCALE", "0.5"))

# ── Estado global del worker ──────────────────────────────────────────────────
pipe: FluxPipeline | None = None
_ready = False
_active_adapter_names: list[str] = []


def log(msg: str):
    """Log con timestamp y flush inmediato."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_gpu_state(label: str = ""):
    """Log detallado del estado GPU para debugging."""
    if not torch.cuda.is_available():
        log(f"   ⚠️  GPU no disponible ({label})")
        return
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory // (1024**3)
        allocated = torch.cuda.memory_allocated() // (1024**3)
        reserved = torch.cuda.memory_reserved() // (1024**3)
        free = total_mem - allocated
        log(f"   💾 GPU ({label}): {allocated}GB/{total_mem}GB | Reserved: {reserved}GB | Free: {free}GB")
    except Exception as e:
        log(f"   ⚠️  Error logging GPU state: {e}")


def free_gpu():
    """Limpia GPU agresivamente."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log(f"   ✅ GPU cleared")
    except Exception as e:
        log(f"   ⚠️  Error freeing GPU: {e}")


_downloaded_loras: dict[str, str] = {}


def download_lora(url: str, filename: str, max_retries: int = 3) -> str:
    """
    Descarga un LoRA a disco. Retorna el path (str). Cachea si ya existe.
    Con reintentos automáticos para descargas fallidas.
    """
    path = os.path.join(LORA_DIR, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) // (1024**2)
        log(f"   📂 {filename} (cacheado: {size_mb}MB)")
        return path
    
    log(f"   📥 Descargando {filename}...")
    t0 = time.time()
    
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            
            size_mb = os.path.getsize(path) // (1024**2)
            elapsed = time.time() - t0
            log(f"   ✅ {filename} ({elapsed:.1f}s, {size_mb}MB)")
            return path
        
        except Exception as e:
            elapsed = time.time() - t0
            log(f"   ⚠️  Attempt {attempt}/{max_retries} failed ({elapsed:.0f}s): {str(e)[:100]}")
            if attempt < max_retries:
                time.sleep(5)
                continue
            raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts: {e}")
    
    return path  # nunca lista — SIEMPRE str


def prepare_loras() -> None:
    """Descarga y cachea las LoRA files antes del primer job."""
    global _downloaded_loras
    os.makedirs(LORA_DIR, exist_ok=True)
    if _downloaded_loras:
        log("   ℹ️  LoRAs ya descargadas")
        return

    log("🔧 Preparando LoRAs antes del primer job...")
    if LORA1_URL:
        _downloaded_loras["kate"] = download_lora(LORA1_URL, "kate_identity.safetensors")
    if LORA2_URL:
        _downloaded_loras["nsfw"] = download_lora(LORA2_URL, "nsfw_style.safetensors")
    log("   ✅ LoRAs preparadas")


def init_pipeline():
    """
    Inicializa el pipeline UNA sola vez (lazy, en el primer job).
    
    Orden CRÍTICO en diffusers 0.30.x:
      1. from_pretrained  → modelo en CPU/bfloat16
      2. load_lora_weights (x2, con adapter_name único cada vez)
      3. set_adapters     → activa ambos con sus escalas
      4. enable_*_cpu_offload → registra los hooks de offload
      
    ❌ PROHIBIDO: fuse_lora(), remove_all_hooks antes del offload
    """
    global pipe, _ready, _active_adapter_names
    if _ready:
        log(f"   ℹ️  Pipeline ya inicializado")
        return

    t0 = time.time()
    log(f"🔄 [{BUILD_ID}] Inicializando pipeline (cold-start)...")
    log_gpu_state("inicio")
    os.makedirs(LORA_DIR, exist_ok=True)

    try:
        # ── 1. Cargar modelo base en CPU (bfloat16) ───────────────────────────────
        log("   📦 Cargando FLUX.1-dev (bfloat16)...")
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN or None,
        )
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            log(f"   GPU: {device_name} ({total_mem}GB VRAM)")
        
        # Optimizaciones de VAE (sin VRAM extra)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            log(f"   ✅ VAE slicing + tiling activado")

        # Reducción adicional de memoria para la inferencia
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            log(f"   ✅ Attention slicing activado")
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log(f"   ✅ XFormers memory efficient attention activado")
            except Exception as ex:
                log(f"   ⚠️  No se pudo activar xformers: {ex}")

        log(f"   ✅ Modelo base cargado ({time.time()-t0:.0f}s)")

        # ── 2. Cargar LoRAs INDIVIDUALMENTE (cada uno recibe UN str path) ─────────
        names:  list[str]   = []
        scales: list[float] = []

        if LORA1_URL:
            lora1_path = _downloaded_loras.get("kate")
            if not lora1_path:
                log(f"   📥 LoRA 1 (kate_identity) desde URL...")
                lora1_path = download_lora(LORA1_URL, "kate_identity.safetensors")
            t1 = time.time()
            pipe.load_lora_weights(lora1_path, adapter_name="kate")
            names.append("kate")
            scales.append(LORA1_SCALE)
            log(f"   ✓ LoRA kate cargado ({time.time()-t1:.1f}s)")

        if LORA2_URL:
            lora2_path = _downloaded_loras.get("nsfw")
            if not lora2_path:
                log(f"   📥 LoRA 2 (nsfw_style) desde URL...")
                lora2_path = download_lora(LORA2_URL, "nsfw_style.safetensors")
            t2 = time.time()
            pipe.load_lora_weights(lora2_path, adapter_name="nsfw")
            names.append("nsfw")
            scales.append(LORA2_SCALE)
            log(f"   ✓ LoRA nsfw cargado ({time.time()-t2:.1f}s)")

        # ── 3. Activar adaptadores con sus escalas ────────────────────────────────
        if names:
            pipe.set_adapters(names, adapter_weights=scales)
            _active_adapter_names = names
            log(f"   🎛️  Adaptadores activos: {names} | escalas: {scales}")
        else:
            log(f"   ⚠️  No LoRAs to load (LORA1_URL, LORA2_URL not set)")

        # ── 4. CPU Offload (DESPUÉS de cargar LoRAs — NUNCA antes) ───────────────
        # IMPORTANTE: enable_*_cpu_offload registra hooks internos de diffusers.
        # Llamar remove_all_hooks() después destruiría esos hooks → error en inferencia.
        # ❌ NUNCA llamar fuse_lora() — es incompatible con sequential_cpu_offload()
        
        log(f"   ⚙️  Activando {OFFLOAD_TYPE} CPU offload...")
        if OFFLOAD_TYPE == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        
        log_gpu_state("post-offload")

        _ready = True
        total_init_time = time.time() - t0
        log(f"✅ [{BUILD_ID}] Pipeline LISTO en {total_init_time:.0f}s")
        log(f"   Config: offload={OFFLOAD_TYPE} | max_side={MAX_SIDE} | steps={DEFAULT_STEPS}")
        log(f"   LORA1_SCALE={LORA1_SCALE} | LORA2_SCALE={LORA2_SCALE}")
        log_gpu_state("pipeline-ready")
    
    except Exception as e:
        log(f"❌ Error inicializando pipeline: {e}")
        traceback.print_exc()
        free_gpu()
        raise


def handler(job):
    """
    Handler RunPod: recibe un job y devuelve imagen en base64.
    Con validación exhaustiva, logging detallado y manejo robusto de errores.
    """
    inp     = job.get("input", {})
    prompt  = inp.get("prompt", "KATESTYLE portrait, soft lighting, masterpiece, best quality")
    width   = min(int(inp.get("width",  MAX_SIDE)), MAX_SIDE)
    height  = min(int(inp.get("height", MAX_SIDE)), MAX_SIDE)
    steps   = int(inp.get("steps",    DEFAULT_STEPS))
    guidance = float(inp.get("guidance", 3.5))
    seed    = inp.get("seed", None)

    job_id = job.get("id", "unknown")
    log(f"\n🎨 Job {job_id} recibido")
    log(f"   Res: {width}x{height} | Steps: {steps} | Guidance: {guidance}")
    log(f"   Prompt: {prompt[:120]}...")
    log_gpu_state("job-start")

    try:
        # Cold-start lazy: solo carga en el primer job
        init_pipeline()

        # ── Validaciones previas ──────────────────────────────────────────────────
        if not pipe:
            raise RuntimeError("Pipeline is None after init_pipeline()")
        
        if width % 16 != 0 or height % 16 != 0:
            log(f"   ⚠️  Redimensionando a múltiplos de 16...")
            width = (width // 16) * 16
            height = (height // 16) * 16
        
        if steps < 1 or steps > 50:
            log(f"   ⚠️  Steps={steps} fuera de rango [1,50], usando {DEFAULT_STEPS}")
            steps = DEFAULT_STEPS

        # ── Generar imagen ────────────────────────────────────────────────────────
        t0 = time.time()
        
        # Crear generator si hay seed
        gen = None
        if seed is not None:
            try:
                gen = torch.Generator("cuda").manual_seed(int(seed))
                log(f"   🌱 Seed: {seed}")
            except Exception as e:
                log(f"   ⚠️  Error setting seed: {e}, ignorando")
        
        # Callback de progreso (compatible con diffusers 0.28+)
        step_times = {"start": time.time(), "last": time.time()}
        def on_step_end(pipeline, i, t, callback_kwargs):
            try:
                n = i + 1
                now = time.time()
                elapsed_step = now - step_times["last"]
                elapsed_total = now - step_times["start"]
                
                # Log cada 4 steps o al final
                if n == 1 or n == steps or n % 4 == 0:
                    eta = (elapsed_step * (steps - n)) if n > 1 else steps * 5
                    log(f"   📊 Step {n:3d}/{steps} ({elapsed_step:.1f}s) | ETA: {eta:.0f}s")
                
                step_times["last"] = now
            except Exception as e:
                log(f"   ⚠️  Error en callback: {e}")
            
            return callback_kwargs

        # Inferencia
        log(f"   ⚙️  Generando...")
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                callback_on_step_end=on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        # ── Post-procesar imagen ──────────────────────────────────────────────────
        img = result.images[0]
        
        # Validar que la imagen sea válida
        if img is None or img.size[0] == 0 or img.size[1] == 0:
            raise RuntimeError(f"Invalid image: {img}")
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        gen_time = round(time.time() - t0, 2)
        img_size_mb = len(b64) // (1024 * 1024)
        
        log(f"✅ Imagen generada en {gen_time}s | {width}x{height} | {img_size_mb}MB")
        log_gpu_state("job-complete")
        
        return {
            "image_base64": b64,
            "format":       "jpeg",
            "width":        width,
            "height":       height,
            "generation_time_seconds": gen_time,
            "build_id":     BUILD_ID,
            "adapters":     _active_adapter_names,
            "seed":         seed,
        }

    except torch.cuda.OutOfMemoryError as oom:
        log(f"❌ CUDA OOM después de {time.time()-t0:.0f}s")
        log_gpu_state("oom")
        free_gpu()
        return {
            "error":    "CUDA out of memory. Reduce width/height a 512x512 o steps a 16. Redimensiona a múltiplos de 16.",
            "build_id": BUILD_ID,
            "job_id":   job_id,
        }
    
    except RuntimeError as rte:
        log(f"❌ RuntimeError: {rte}")
        traceback.print_exc()
        free_gpu()
        return {
            "error": f"Runtime error: {str(rte)[:200]}",
            "build_id": BUILD_ID,
            "job_id": job_id,
        }
    
    except Exception as exc:
        log(f"❌ Error inesperado: {exc}")
        traceback.print_exc()
        log_gpu_state("error")
        free_gpu()
        return {
            "error": f"Unexpected error: {str(exc)[:200]}",
            "build_id": BUILD_ID,
            "job_id": job_id,
        }


# ── CRÍTICO: el servidor RunPod DEBE arrancar antes que el modelo ─────────────
# El modelo se carga en el primer job (lazy init), no al iniciar el container.
# Esto evita que los jobs se queden en cola mientras el modelo carga.
log("")
log("=" * 70)
log(f"🚀 [{BUILD_ID}] Servidor RunPod iniciado")
log(f"   Model: {MODEL_ID}")
log(f"   Offload: {OFFLOAD_TYPE} | MAX_SIDE: {MAX_SIDE} | DEFAULT_STEPS: {DEFAULT_STEPS}")
log(f"   LORA1: {LORA1_SCALE} scale | LORA2: {LORA2_SCALE} scale")
log(f"   El modelo cargará en el PRIMER job (lazy initialization)")
log(f"   Pero las LoRAs ya se descargarán al iniciar el worker")
log("=" * 70)
log("")

try:
    prepare_loras()
    runpod.serverless.start({"handler": handler})
except KeyboardInterrupt:
    log("⚠️  Servidor detenido por usuario")
    free_gpu()
    sys.exit(0)
except Exception as e:
    log(f"❌ Error iniciando servidor: {e}")
    traceback.print_exc()
    sys.exit(1)
