#!/usr/bin/env python3
"""
=============================================================================
Pixel_Kate — Generación NSFW vía RunPod Serverless (FLUX + Dual LoRA)
=============================================================================
USO:
    python3 generate_nsfw_runpod.py --prompt "KATESTYLE, nude, artistic"
    python3 generate_nsfw_runpod.py --prompt "..." --seed 42 --steps 30
=============================================================================
"""

import os
import sys
import time
import base64
import argparse
import requests

# ── Config — Rellenar después de crear el endpoint en RunPod ────────────────
RUNPOD_API_KEY   = os.environ.get("RUNPOD_API_KEY", "")   # Tu API Key de RunPod
RUNPOD_ENDPOINT  = os.environ.get("RUNPOD_ENDPOINT", "")  # ID del endpoint (ej: abc123xyz)

# URLs de los LoRAs (ya las tenemos desde antes)
KATE_LORA_URL    = "https://v3b.fal.media/files/b/0a9aca2a/hToqwVnqjQOWYiHgKNx9O_pytorch_lora_weights.safetensors"
NSFW_LORA_URL    = "https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors"
# ─────────────────────────────────────────────────────────────────────────────


def call_runpod(payload: dict, timeout_sec: int = 300) -> dict:
    """
    Envía un job a RunPod Serverless y espera el resultado.
    Usa /runsync para esperar hasta que la imagen esté lista.
    """
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
        print("❌ ERROR: Falta configurar RUNPOD_API_KEY y RUNPOD_ENDPOINT")
        print("   Edita el archivo o exporta las variables de entorno:")
        print("   export RUNPOD_API_KEY='tu_key_aqui'")
        print("   export RUNPOD_ENDPOINT='tu_endpoint_id_aqui'")
        sys.exit(1)

    url = f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"\n📤 Enviando job a RunPod...")
    print(f"   Endpoint: {RUNPOD_ENDPOINT}")

    resp = requests.post(url, json={"input": payload}, headers=headers, timeout=timeout_sec)

    if resp.status_code != 200:
        print(f"\n❌ Error HTTP {resp.status_code}:")
        print(f"   {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    status = data.get("status")

    if status == "FAILED":
        print(f"\n❌ El job falló en RunPod:")
        print(f"   {data.get('error', 'Sin detalle')}")
        sys.exit(1)

    if status != "COMPLETED":
        print(f"\n⚠️  Estado inesperado: {status}")
        print(data)
        sys.exit(1)

    return data.get("output", {})


def save_image(output: dict, output_dir: str = "./test_outputs") -> str:
    """Decodifica la imagen base64 y la guarda en disco."""
    img_b64 = output.get("image_base64")
    if not img_b64:
        print(f"❌ No se encontró imagen en la respuesta.")
        print(f"   Respuesta completa: {output}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"nsfw_runpod_{timestamp}.jpg")

    img_bytes = base64.b64decode(img_b64)
    with open(filename, "wb") as f:
        f.write(img_bytes)

    return filename


def main():
    parser = argparse.ArgumentParser(description="Genera imágenes NSFW con RunPod + FLUX + Dual LoRA")
    parser.add_argument("--prompt",
                        default="KATESTYLE, beautiful portrait, soft lighting, high quality photography",
                        help="Prompt de generación (incluir KATESTYLE)")
    parser.add_argument("--lora1-url", default=KATE_LORA_URL,
                        help="URL directa al .safetensors del LoRA de Kate")
    parser.add_argument("--lora1-scale", type=float, default=0.85,
                        help="Intensidad del LoRA de Kate (0.5-1.0)")
    parser.add_argument("--lora2-url", default=NSFW_LORA_URL,
                        help="URL directa al .safetensors del LoRA NSFW")
    parser.add_argument("--lora2-scale", type=float, default=0.5,
                        help="Intensidad del LoRA NSFW (0.3-0.7)")
    parser.add_argument("--width",  type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps",  type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="./test_outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("🎨 Pixel Kate — NSFW Generator (RunPod Serverless)")
    print("=" * 60)
    print(f"   Prompt: {args.prompt}")
    print(f"   LoRA Kate: scale={args.lora1_scale}")
    print(f"   LoRA NSFW: scale={args.lora2_scale}")

    # Construir payload para el worker
    payload = {
        "prompt": args.prompt,
        "lora1_url": args.lora1_url,
        "lora1_scale": args.lora1_scale,
        "lora2_url": args.lora2_url,
        "lora2_scale": args.lora2_scale,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
    }
    if args.seed:
        payload["seed"] = args.seed

    t0 = time.time()
    output = call_runpod(payload)
    elapsed = time.time() - t0

    gen_time = output.get("generation_time_seconds", "?")
    loras    = output.get("loras_used", [])
    print(f"\n✅ Job completado en {elapsed:.1f}s total")
    print(f"   Tiempo de generación GPU: {gen_time}s")
    print(f"   LoRAs aplicados: {loras}")

    filename = save_image(output, args.output_dir)
    print(f"\n🖼️  Imagen guardada en: {filename}")


if __name__ == "__main__":
    main()
