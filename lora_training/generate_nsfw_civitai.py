#!/usr/bin/env python3
"""
=============================================================================
Pixel_Kate — Generación NSFW vía Civitai Orchestration API v2 (sin SDK)
=============================================================================
El SDK civitai-py no tiene soporte para Flux, así que usamos la API REST v2
directamente, que es la forma correcta y documentada oficialmente.

USO:
    python3 generate_nsfw_civitai.py
    python3 generate_nsfw_civitai.py --prompt "KATESTYLE, nude, artistic" --lora2-urn "urn:air:flux1:lora:civitai:XXXXX@YYYYY"
=============================================================================
"""

import os
import sys
import time
import argparse
import requests
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────
CIVITAI_TOKEN   = "9e5415ae2d027500a4640e81c4ffa674"
KATE_LORA_URN   = "urn:air:flux1:lora:civitai:2636543@2960252"  # LoRA privado de Kate
FLUX_DEV_URN    = "urn:air:flux1:diffuser:civitai:618692@691639" # Flux1-Dev oficial
# NOTA: El mismo endpoint soporta NSFW con Yellow Buzz (comprar en civitai.red)
API_BASE        = "https://orchestration.civitai.com"
WORKFLOWS_URL   = f"{API_BASE}/v2/consumer/workflows"
# ───────────────────────────────────────────────────────────────────────────


def build_headers():
    return {
        "Authorization": f"Bearer {CIVITAI_TOKEN}",
        "Content-Type": "application/json"
    }


def submit_job(prompt: str, lora1_urn: str, lora2_urn: str = "", lora2_scale: float = 0.55,
               width: int = 1024, height: int = 1024) -> str:
    """
    Envía el trabajo a la Orchestration API v2 de Civitai.
    Retorna el workflowId para hacer polling.
    """
    # Construir lista de LoRAs (estructura documentada oficial para Flux1)
    loras = {lora1_urn: 0.85}
    if lora2_urn:
        loras[lora2_urn] = lora2_scale

    payload = {
        "allowMatureContent": True,   # ← Habilitar contenido adulto/NSFW
        "steps": [
            {
                "$type": "imageGen",
                "input": {
                    "engine": "sdcpp",
                    "ecosystem": "flux1",
                    "operation": "createImage",
                    "diffuserModel": FLUX_DEV_URN,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": 25,
                    "cfgScale": 1,
                    "guidance": 3.5,
                    "nsfw": True,         # ← Flag adicional de contenido explícito
                    "loras": loras
                }
            }
        ]
    }

    print(f"\n📤 Payload enviado:\n   Prompt: {prompt[:80]}...")
    print(f"   LoRAs: {list(loras.keys())}")
    print(f"   Endpoint: {WORKFLOWS_URL}")

    resp = requests.post(WORKFLOWS_URL, json=payload, headers=build_headers(), timeout=30)

    if resp.status_code not in [200, 201, 202]:
        print(f"\n❌ Error {resp.status_code} al enviar el trabajo:")
        print(f"   {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    workflow_id = data.get("id") or data.get("workflowId") or data.get("token")
    if not workflow_id:
        print(f"\n❌ No se recibió un ID de trabajo. Respuesta completa:")
        print(data)
        sys.exit(1)

    print(f"\n✅ Trabajo creado. ID: {workflow_id}")
    return workflow_id, data


def poll_job(workflow_id: str, timeout_sec: int = 300) -> str:
    """
    Hace polling cada 5 segundos hasta que el trabajo termine.
    Retorna la URL de la imagen.
    """
    status_url = f"{WORKFLOWS_URL}/{workflow_id}"
    print(f"⏳ Esperando resultado (hasta {timeout_sec}s)...")

    elapsed = 0
    while elapsed < timeout_sec:
        resp = requests.get(status_url, headers=build_headers(), timeout=30)
        data = resp.json()

        status = data.get("status") or data.get("state") or "unknown"
        print(f"   [{elapsed}s] Estado: {status}")

        if status.lower() in ["succeeded", "completed", "done"]:
            # Buscar la URL de la imagen en la respuesta
            steps = data.get("steps", [])
            for step in steps:
                output = step.get("output", {})
                images = output.get("images", [])
                for img in images:
                    url = img.get("url") or img.get("blobUrl")
                    if url:
                        return url
            # Fallback: buscar en estructura plana
            result = data.get("result", {})
            url = result.get("blobUrl") or result.get("url")
            if url:
                return url
            print("⚠️  Trabajo completado pero no se encontró URL. Respuesta completa:")
            print(data)
            sys.exit(1)

        elif status.lower() in ["failed", "cancelled", "rejected", "error"]:
            print(f"\n❌ El trabajo falló. Estado: {status}")
            print(f"   Detalles: {data}")
            sys.exit(1)

        time.sleep(5)
        elapsed += 5

    print(f"\n❌ Timeout: el trabajo no completó en {timeout_sec} segundos.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Genera imágenes NSFW con Civitai API v2 (FLUX)")
    parser.add_argument("--lora1-urn", default=KATE_LORA_URN,
                        help="URN del LoRA de Kate")
    parser.add_argument("--lora2-urn", default="",
                        help="URN de un segundo LoRA NSFW (opcional)")
    parser.add_argument("--lora2-scale", type=float, default=0.55)
    parser.add_argument("--prompt",
                        default="KATESTYLE, beautiful portrait, soft lighting, high quality photography")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output-dir", default="./test_outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("🎨 Generación NSFW — Civitai API v2 (Flux1)")
    print("=" * 60)

    # Enviar trabajo
    workflow_id, raw = submit_job(
        prompt=args.prompt,
        lora1_urn=args.lora1_urn,
        lora2_urn=args.lora2_urn,
        lora2_scale=args.lora2_scale,
        width=args.width,
        height=args.height
    )

    # Esperar resultado
    image_url = poll_job(workflow_id)

    # Descargar
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"nsfw_{timestamp}.jpg")
    print(f"\n📥 Descargando imagen desde:\n   {image_url}")
    urllib.request.urlretrieve(image_url, filename)
    print(f"✅ Guardado en: {filename}")


if __name__ == "__main__":
    main()
