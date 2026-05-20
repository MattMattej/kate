# =============================================================================
# Pixel_Kate — n8n V1 (TikTok + Imágenes)
# Imagen mínima: solo n8n + curl para healthcheck
# Toda la generación de imágenes se hace via API (fal.ai HTTP calls)
# =============================================================================

FROM n8nio/n8n:1.123.42

USER root

# Solo lo mínimo necesario
RUN apk add --no-cache curl bash && rm -rf /var/cache/apk/*

USER node

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5678/healthz || exit 1

EXPOSE 5678
