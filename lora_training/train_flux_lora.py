#!/usr/bin/env python3
"""
=============================================================================
Pixel_Kate — FLUX.1 LoRA Training via fal.ai
Entrena un LoRA personalizado de Kate usando FLUX.1 [dev]
=============================================================================

PREREQUISITOS:
    pip install fal-client requests python-dotenv

USO:
    export FAL_KEY="tu_api_key_de_fal_ai"
    python3 train_flux_lora.py --zip ./kate_images.zip --name "pixel_kate_v1"

IMÁGENES RECOMENDADAS:
    - 20-30 fotos de Kate
    - Resolución 1024x1024 px (recortar con Birme.net)
    - Variedad: diferentes ángulos, expresiones, ropa, iluminación
    - Formato: JPG o PNG
    - Comprimir en ZIP antes de ejecutar
=============================================================================
"""

import os
import sys
import argparse
import json
import time
import fal_client


def upload_dataset(zip_path: str) -> str:
    """Sube el ZIP de imágenes a fal.ai y retorna la URL."""
    print(f"📤 Subiendo dataset: {zip_path}")
    
    url = fal_client.upload_file(zip_path)
    
    print(f"✅ Dataset subido: {url}")
    return url


def train_flux_lora(
    dataset_url: str,
    trigger_word: str = "KATESTYLE",
    steps: int = 1500,
    lora_rank: int = 16,
    model_name: str = "pixel_kate_v1"
) -> dict:
    """
    Inicia el entrenamiento LoRA en fal.ai usando FLUX.1 [dev].
    
    Parámetros:
        dataset_url   : URL del ZIP de imágenes (retornada por upload_dataset)
        trigger_word  : Palabra que activará el estilo de Kate en prompts
        steps         : Pasos de entrenamiento (1000-2000 recomendado)
        lora_rank     : Rango del LoRA (16 = balance calidad/tamaño, 32 = más calidad)
        model_name    : Nombre para identificar el modelo entrenado
    
    Retorna:
        dict con la URL del modelo LoRA (.safetensors)
    """
    print(f"\n🚀 Iniciando entrenamiento LoRA FLUX.1 [dev]...")
    print(f"   Trigger word: {trigger_word}")
    print(f"   Steps: {steps}")
    print(f"   LoRA Rank: {lora_rank}")
    print(f"   Modelo: {model_name}")
    print(f"\n⏳ Esto puede tardar 20-40 minutos...")

    def on_queue_update(update):
        if hasattr(update, 'logs'):
            for log in update.logs:
                print(f"   📊 {log['message']}")

    result = fal_client.subscribe(
        "fal-ai/flux-lora-fast-training",
        arguments={
            "images_data_url": dataset_url,
            "trigger_word": trigger_word,
            "steps": steps,
            "lora_rank": lora_rank,
            "multiresolution_training": True,
            "create_masks": True,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    print(f"\n✅ ¡Entrenamiento completado!")
    print(f"   Resultado: {json.dumps(result, indent=2)}")
    
    return result


def save_training_result(result: dict, model_name: str):
    """Guarda el resultado del entrenamiento en un archivo JSON."""
    output_file = f"{model_name}_training_result.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "model_name": model_name,
            "training_result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "instructions": {
                "trigger_word": "Usa el trigger word en tus prompts para activar el estilo",
                "example_prompt": "photo of KATESTYLE, beautiful woman, high quality, realistic",
                "api_endpoint": "fal-ai/flux-lora",
                "how_to_use": "Incluir la URL de config_file en llamadas a la API de generación"
            }
        }, f, indent=2)
    
    print(f"\n💾 Resultado guardado en: {output_file}")
    print(f"   ⚠️ Guarda este archivo — contiene la URL de tu modelo LoRA")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un LoRA de FLUX.1 en fal.ai para Pixel_Kate"
    )
    parser.add_argument(
        "--zip", required=False,
        help="Ruta al archivo ZIP con las imágenes de Kate (20-30 fotos, 1024x1024)"
    )
    parser.add_argument(
        "--url", required=False,
        help="URL pública del dataset (para evitar fallos de subida 403)"
    )
    parser.add_argument(
        "--name", default="pixel_kate_v1",
        help="Nombre del modelo (default: pixel_kate_v1)"
    )
    parser.add_argument(
        "--trigger", default="KATESTYLE",
        help="Palabra activadora en prompts (default: KATESTYLE)"
    )
    parser.add_argument(
        "--steps", type=int, default=1500,
        help="Pasos de entrenamiento (default: 1500)"
    )
    parser.add_argument(
        "--rank", type=int, default=16,
        help="Rango del LoRA (default: 16, max: 32)"
    )
    args = parser.parse_args()

    # Verificar API key
    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        print("❌ ERROR: Define FAL_KEY como variable de entorno")
        print("   export FAL_KEY=tu_api_key_de_fal_ai")
        sys.exit(1)

    if not args.zip and not args.url:
        print("❌ ERROR: Debes proporcionar --zip o --url")
        sys.exit(1)

    # Verificar que el ZIP existe si se usa --zip
    if args.zip and not os.path.exists(args.zip):
        print(f"❌ ERROR: No se encontró el archivo ZIP: {args.zip}")
        sys.exit(1)

    print("=" * 60)
    print("🎨 Pixel_Kate — FLUX LoRA Trainer")
    print("=" * 60)

    # 1. Obtener dataset URL
    if args.url:
        dataset_url = args.url
        print(f"✅ Usando dataset desde URL pública: {dataset_url}")
    else:
        dataset_url = upload_dataset(args.zip)

    # 2. Entrenar
    result = train_flux_lora(
        dataset_url=dataset_url,
        trigger_word=args.trigger,
        steps=args.steps,
        lora_rank=args.rank,
        model_name=args.name
    )

    # 3. Guardar resultado
    save_training_result(result, args.name)

    print("\n" + "=" * 60)
    print("✅ ¡LoRA entrenado con éxito!")
    print("   Siguiente paso: ejecutar generate_test_image.py para probar")
    print("=" * 60)


if __name__ == "__main__":
    main()
