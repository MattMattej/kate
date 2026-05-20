#!/usr/bin/env python3
import os
import sys
import time
import threading
import fal_client

is_uploading = True

def upload_spinner():
    chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    start_time = time.time()
    while is_uploading:
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\r   🚀 Subiendo a Fal.ai... {chars[i % len(chars)]} (Tiempo: {elapsed}s) (Dependiendo de tu conexión puede demorar unos minutos)")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

def main():
    local_filename = "flux-nsfw-uncensored.safetensors"
    
    if not os.path.exists(local_filename):
        print("============================================================")
        print("❌ El archivo no se encontró localmente.")
        print("La descarga por consola falló por un corte de conexión.")
        print("============================================================")
        print("\nPor favor, DESCARGA EL ARCHIVO MANUALMENTE DESDE TU NAVEGADOR:")
        print("👉 https://huggingface.co/Heartsync/Flux-NSFW-uncensored/resolve/main/lora.safetensors")
        print(f"\nUna vez descargado, renómbralo a '{local_filename}'")
        print("y colócalo en la carpeta 'pixel_kate/lora_training' (donde estás ahora).")
        print("\nLuego, vuelve a ejecutar este script para subirlo.")
        return
        
    print(f"\n🚀 Iniciando subida de {local_filename} a fal.ai...")
    
    global is_uploading
    is_uploading = True
    spinner_thread = threading.Thread(target=upload_spinner)
    spinner_thread.start()
    
    try:
        # fal_client utiliza la variable de entorno FAL_KEY
        fal_url = fal_client.upload_file(local_filename)
        is_uploading = False
        spinner_thread.join()
        
        print("\n\n✅ ¡Subida exitosa a Fal.ai!")
        print("Usa esta URL en tu script:")
        print("-" * 60)
        print(fal_url)
        print("-" * 60)
    except Exception as e:
        is_uploading = False
        spinner_thread.join()
        print(f"\n\n❌ Error al subir: {e}")

if __name__ == "__main__":
    if not os.environ.get("FAL_KEY"):
        os.environ["FAL_KEY"] = "b23cfae9-852e-496c-b44c-df34a5efc788:db5b9c3885f9954046035274f4c56f58"
    main()
