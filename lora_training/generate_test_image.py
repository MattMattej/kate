#!/usr/bin/env python3
"""
=============================================================================
Pixel_Kate — Generación con dos LoRAs (identidad + estilo/uncensored)
=============================================================================
USO:
    export FAL_KEY="b23cfae9-852e-496c-b44c-df34a5efc788:db5b9c3885f9954046035274f4c56f58"
    python3 generate_test_image.py \
        --lora1-url "AQUI_TU_URL_DE_IDENTIDAD" \
        --lora2-url "https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors" \
        --trigger "KATESTYLE" \
        --prompt "full body photo of KATESTYLE, nude, artistic, soft lighting"
=============================================================================
"""

import os
import sys
import argparse
import urllib.request
import time
import fal_client


def generate_image(
    lora1_url: str,
    lora2_url: str = None,
    lora1_scale: float = 1.0,
    lora2_scale: float = 0.7,
    prompt: str = "",
    trigger_word: str = "KATESTYLE",
    negative_prompt: str = "blurry, low quality, distorted face, bad anatomy",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    num_images: int = 4,
    seed: int = None
) -> list:
    """
    Genera imágenes usando dos LoRAs:
        - LoRA1: identidad personal
        - LoRA2: uncensored / desbloqueo de contenido / estilo
    """
    # Asegurar trigger word en prompt
    if trigger_word and trigger_word not in prompt:
        prompt = f"{trigger_word}, {prompt}"

    print(f"\n🎨 Generando {num_images} imagen(es)...")
    print(f"   Prompt: {prompt}")
    print(f"   LoRA1 (identidad): {lora1_url[:60]}... escala={lora1_scale}")
    if lora2_url:
        print(f"   LoRA2 (estilo): {lora2_url[:60]}... escala={lora2_scale}")

    # Construir lista de LoRAs
    loras_list = [{"path": lora1_url, "scale": lora1_scale}]
    if lora2_url:
        loras_list.append({"path": lora2_url, "scale": lora2_scale})

    def on_queue_update(update):
        if hasattr(update, 'logs') and update.logs:
            for log in update.logs:
                print(f"   [FAL LOG] {log.get('message', log)}")
        elif hasattr(update, 'status'):
            print(f"   [FAL STATUS] {update.status}")

    print("\n⏳ Enviando petición a fal.ai (Si es la primera vez que se usa el LoRA 2, tardará 1-2 minutos en descargarlo)...")

    result = fal_client.subscribe(
        "fal-ai/flux/dev",
        arguments={
            "prompt": prompt,
            "negative_prompt": negative_prompt, 
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": False,   # ← CRUCIAL para contenido adulto
            "loras": loras_list,              # ← ahora con dos LoRAs
            **({"seed": seed} if seed else {})
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    images = result.get("images", [])
    print(f"\n✅ {len(images)} imagen(es) generada(s)")
    return images


def download_images(images: list, output_dir: str = "./test_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(images):
        url = img.get("url")
        if not url:
            continue
        filename = os.path.join(output_dir, f"two_loras_{timestamp}_{i+1}.jpg")
        print(f"   📥 Descargando: {filename}")
        urllib.request.urlretrieve(url, filename)
        print(f"   ✅ Guardado: {filename}")
    print(f"\n📁 Imágenes guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera imágenes con dos LoRAs (identidad + uncensored/estilo)"
    )
    parser.add_argument("--lora1-url", required=True,
                        help="URL del LoRA de identidad (tu rostro)")
    parser.add_argument("--lora2-url", 
                        default="https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors",
                        help="URL del LoRA secundario (HuggingFace / Fal)")
    parser.add_argument("--lora1-scale", type=float, default=1.0,
                        help="Intensidad del LoRA de identidad (0.5-1.2)")
    parser.add_argument("--lora2-scale", type=float, default=0.7,
                        help="Intensidad del LoRA secundario (0.5-1.0)")
    parser.add_argument("--trigger", default="KATESTYLE",
                        help="Trigger word del LoRA de identidad")
    parser.add_argument("--prompt", 
                        default="full body shot, portrait of a beautiful woman, high quality photo",
                        help="Prompt (se añade trigger automáticamente)")
    parser.add_argument("--guidance", type=float, default=3.5,
                        help="Fuerza del prompt (Guidance scale). Bajar a 3.0 o 2.5 si hay imágenes negras.")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--output-dir", default="./test_outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        # Asignamos la FAL_KEY directamente si no está en el entorno, según lo solicitado
        os.environ["FAL_KEY"] = "b23cfae9-852e-496c-b44c-df34a5efc788:db5b9c3885f9954046035274f4c56f58"
        print("ℹ️ FAL_KEY cargada automáticamente en el script.")

    print("=" * 60)
    print("🎨 Generación con dos LoRAs (identidad + estilo)")
    print("=" * 60)

    images = generate_image(
        lora1_url=args.lora1_url,
        lora2_url=args.lora2_url,
        lora1_scale=args.lora1_scale,
        lora2_scale=args.lora2_scale,
        prompt=args.prompt,
        trigger_word=args.trigger,
        guidance_scale=args.guidance,
        num_images=args.num_images,
        seed=args.seed
    )

    download_images(images, args.output_dir)

    print("\n✅ Prueba completada.")
    print("💡 Si la identidad se ve alterada, reduce lora2_scale (ej. 0.5).")
    print("💡 Si el filtro de desnudo sigue activo, revisa que enable_safety_checker=False.")


if __name__ == "__main__":
    main()
