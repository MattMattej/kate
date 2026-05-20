#!/usr/bin/env python3
"""
=============================================================================
Pixel_Kate — Generación SFW (Solo Identidad Kate) vía Fal.ai
=============================================================================
USO:
    python3 generate_sfw_fal.py \
        --lora-url "https://v3b.fal.media/files/.../kate.safetensors" \
        --prompt "full body photo of KATESTYLE, wearing a red dress, smiling, daytime"
=============================================================================
"""

import os
import argparse
import urllib.request
import time
import fal_client


def generate_image(
    lora_url: str,
    lora_scale: float = 1.0,
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
    if trigger_word and trigger_word not in prompt:
        prompt = f"{trigger_word}, {prompt}"

    print(f"\n🎨 Generando {num_images} imagen(es) SFW...")
    print(f"   Prompt: {prompt}")
    print(f"   LoRA (identidad): {lora_url[:60]}... escala={lora_scale}")

    loras_list = [{"path": lora_url, "scale": lora_scale}]

    def on_queue_update(update):
        if hasattr(update, 'logs') and update.logs:
            for log in update.logs:
                print(f"   [FAL LOG] {log.get('message', log)}")
        elif hasattr(update, 'status'):
            print(f"   [FAL STATUS] {update.status}")

    print("\n⏳ Enviando petición a fal.ai...")

    result = fal_client.subscribe(
        "fal-ai/flux/dev",
        arguments={
            "prompt": prompt,
            "negative_prompt": negative_prompt, 
            "image_size": {"width": width, "height": height},
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": True, # Activado, ya que esto es SFW
            "loras": loras_list,
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
        filename = os.path.join(output_dir, f"sfw_{timestamp}_{i+1}.jpg")
        print(f"   📥 Descargando: {filename}")
        urllib.request.urlretrieve(url, filename)
        print(f"   ✅ Guardado: {filename}")
    print(f"\n📁 Imágenes guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Genera imágenes SFW con Fal.ai")
    parser.add_argument("--lora-url", required=True, help="URL del LoRA de identidad")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Intensidad (0.5-1.2)")
    parser.add_argument("--trigger", default="KATESTYLE", help="Trigger word")
    parser.add_argument("--prompt", default="portrait of KATESTYLE, beautiful photo")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--output-dir", default="./test_outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("FAL_KEY"):
        os.environ["FAL_KEY"] = "b23cfae9-852e-496c-b44c-df34a5efc788:db5b9c3885f9954046035274f4c56f58"

    images = generate_image(
        lora_url=args.lora_url, lora_scale=args.lora_scale,
        prompt=args.prompt, trigger_word=args.trigger,
        guidance_scale=args.guidance, num_images=args.num_images, seed=args.seed
    )
    download_images(images, args.output_dir)

if __name__ == "__main__":
    main()
