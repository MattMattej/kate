# RunPod pixel-kate-nsfw — Configuración definitiva

## Problema que tenías

1. **Imagen Docker vieja** en RunPod (logs dicen `Fusionando... solo cold start` y `Configurando LoRAs una vez por worker`) — **NO** es el código v5.
2. **`FUSE_LORAS=1`** en variables de RunPod → fusión 15–30 min, worker nunca acepta jobs.
3. **`Max workers = 5`** → 5 GPUs arrancan, 4 throttled, 1 atascada fusionando.
4. Job **30 min en IN_QUEUE** porque el worker no llamaba `runpod.serverless.start()` hasta terminar la fusión.

## Checklist RunPod (hacer EN ORDEN)

### 1. Endpoint Settings

| Campo | Valor |
|-------|--------|
| Container image | `mateomatt/pixel-kate-worker:latest` |
| **Max workers** | **1** |
| Active workers | 0 |
| **Execution timeout** | **900** |
| Idle timeout | 300 |
| GPU | RTX 4090 24GB |

### 2. Environment (borrar FUSE_LORAS=1 si existe)

```env
CPU_OFFLOAD_TYPE=model
MAX_IMAGE_SIDE=512  # alias: MAX_SIDE (set to 1024 for 1024x1024 output if your GPU/memory can handle it)
DEFAULT_STEPS=16
FUSE_LORAS=0
LORA1_URL=https://v3b.fal.media/files/b/0a9aca2a/hToqwVnqjQOWYiHgKNx9O_pytorch_lora_weights.safetensors
LORA2_URL=https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors
LORA1_SCALE=0.85
LORA2_SCALE=0.5
```

### 3. Deploy imagen nueva

```bash
git add runpod_worker/ lora_training/generate_nsfw_runpod.py docs/
git commit -m "RunPod v5: server first, no fuse, BUILD_ID verify"
git push origin main
```

Espera GitHub Actions → luego en RunPod: **Release Endpoint** (forzar pull).

### 4. Verificar en logs (OBLIGATORIO)

Tras el deploy, al arrancar un worker debe aparecer:

```text
🚀 [pixel-kate-v5-20260521] Servidor RunPod listo (modelo carga en job 1)
   OFFLOAD=model | MAX=512 | STEPS=16 | FUSE=0
```

Si ves `Fusionando LoRAs en GPU/CPU` o `Configurando LoRAs (una vez por worker)` → **sigues con imagen vieja**.

En el primer job:

```text
🔄 [pixel-kate-v5-20260521] Inicializando (primer job)...
✅ [pixel-kate-v5-20260521] Listo en ~180s
🎨 Generando 512x512 x16 | model
📊 4/16 ... 16/16
✅ Imagen en ~60s
```

## Tiempos esperados

| Fase | Tiempo |
|------|--------|
| IN_QUEUE → IN_PROGRESS | **< 60 s** |
| Primer job (FLUX + LoRAs + imagen) | **~4–6 min** |
| Jobs siguientes (worker caliente) | **~1–2 min** |

## Probar

```bash
cd lora_training
python generate_nsfw_runpod.py --prompt "KATESTYLE portrait of a beautiful woman, soft lighting, masterpiece"
```

## Network Volume (recomendado)

Montar volumen en `/workspace` → cache HF entre reinicios (FLUX no se re-descarga).
