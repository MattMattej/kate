# Instrucciones para publicar el worker en Docker Hub
# Ejecutar desde la carpeta raíz del proyecto (pixel_kate/)

# 1. Construir la imagen Docker
docker build -t TU_USUARIO_DOCKERHUB/pixel-kate-worker:latest ./runpod_worker/

# 2. Subir al Docker Hub
docker push TU_USUARIO_DOCKERHUB/pixel-kate-worker:latest
