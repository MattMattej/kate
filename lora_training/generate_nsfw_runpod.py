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

# Cargar variables de entorno desde el archivo .env de forma manual para evitar dependencias
def load_env():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):  # buscar hasta 5 niveles arriba
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            break
        current_dir = os.path.dirname(current_dir)

load_env()

RUNPOD_API_KEY   = os.environ.get("RUNPOD_API_KEY", "")   # Tu API Key de RunPod
RUNPOD_ENDPOINT  = os.environ.get("RUNPOD_ENDPOINT", "")  # ID del endpoint (ej: abc123xyz)

# URLs de los LoRAs (ya las tenemos desde antes)
KATE_LORA_URL    = "https://v3b.fal.media/files/b/0a9aca2a/hToqwVnqjQOWYiHgKNx9O_pytorch_lora_weights.safetensors"
NSFW_LORA_URL    = "https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors"
# ─────────────────────────────────────────────────────────────────────────────


def call_runpod(payload: dict, timeout_sec: int = 1200) -> dict:
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

    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/runsync"
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
    job_id = data.get("id")
    status = data.get("status")

    # Si entra en cola o está en progreso (común durante arranques en frío/cold start),
    # entramos en un bucle de sondeo para esperar el resultado.
    if status in ["IN_QUEUE", "IN_PROGRESS"]:
        print(f"\n⏳ Esperando worker (cold start + FLUX + LoRAs puede tardar 10-15 min la 1ª vez)...")
        print(f"   Job ID: {job_id}")
        print(f"   Timeout cliente: {timeout_sec}s — en RunPod sube 'Execution Timeout' a ≥900s")

        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/{job_id}"
        poll_start = time.time()

        while status in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = int(time.time() - poll_start)
            if elapsed > timeout_sec:
                print(f"\n❌ Timeout local tras {elapsed}s. Revisa logs en RunPod.")
                sys.exit(1)
            time.sleep(10)
            status_resp = requests.get(status_url, headers=headers, timeout=30)

            if status_resp.status_code != 200:
                print(f"\n❌ Error al consultar estado: HTTP {status_resp.status_code}")
                sys.exit(1)

            data = status_resp.json()
            status = data.get("status")
            print(f"   [{elapsed}s] Estado: {status}")

    if status == "FAILED":
        print(f"\n❌ El job falló en RunPod:")
        print(f"   {data.get('error', 'Sin detalle')}")
        sys.exit(1)

    if status != "COMPLETED":
        print(f"\n⚠️  Estado inesperado final: {status}")
        print(data)
        sys.exit(1)

    output = data.get("output", {})
    if output.get("error"):
        print(f"\n❌ El worker devolvió error:")
        print(f"   {output['error']}")
        sys.exit(1)

    return output


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
    parser.add_argument("--width",  type=int, default=768,
                        help="768 recomendado con sequential offload en GPU 24GB")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps",  type=int, default=24)
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Segundos máximos de espera (default 1200)")
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
    output = call_runpod(payload, timeout_sec=args.timeout)
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
