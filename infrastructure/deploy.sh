#!/bin/bash
# =============================================================================
# Pixel_Kate — Cloud Run Deploy (V1: TikTok + Imágenes)
# DB: Neon PostgreSQL (serverless, gratis)
# Ejecutar DESPUÉS de: service_account_setup.sh, secrets_setup.sh
# =============================================================================

set -e

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-TU_PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="pixelkate-n8n"
IMAGE_NAME="docker.io/n8nio/n8n"
IMAGE_TAG="1.123.42"
SA_NAME="n8n-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME="pixelkate-media-${PROJECT_ID}"

# ── Neon DB ───────────────────────────────────────────────────────────────────
NEON_DB_HOST="${NEON_DB_HOST:-ep-old-block-ajv8cbxf-pooler.c-3.us-east-2.aws.neon.tech}"
NEON_DB_NAME="${NEON_DB_NAME:-neondb}"
NEON_DB_USER="${NEON_DB_USER:-neondb_owner}"

if [ "$PROJECT_ID" = "TU_PROJECT_ID" ]; then
    echo "❌ ERROR: Define PROJECT_ID"
    echo "   export PROJECT_ID=mi-proyecto-gcp"
    exit 1
fi

echo "🚀 Desplegando Pixel_Kate n8n en Cloud Run..."
echo "   Proyecto:  $PROJECT_ID"
echo "   Región:    $REGION"
echo "   Servicio:  $SERVICE_NAME"
echo "   DB Host:   $NEON_DB_HOST"

# ── Paso 1: Desplegar en Cloud Run ────────────────────────────────────────────
echo ""
echo "📦 [1/2] Desplegando en Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="${IMAGE_NAME}:${IMAGE_TAG}" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --service-account="$SA_EMAIL" \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=5 \
    --timeout=300 \
    --port=5678 \
    --set-env-vars="\
DB_TYPE=postgresdb,\
DB_POSTGRESDB_HOST=${NEON_DB_HOST},\
DB_POSTGRESDB_DATABASE=${NEON_DB_NAME},\
DB_POSTGRESDB_USER=${NEON_DB_USER},\
DB_POSTGRESDB_PORT=5432,\
DB_POSTGRESDB_CONNECTION_TIMEOUT=30000,\
DB_POSTGRESDB_SSL_ENABLED=true,\
DB_POSTGRESDB_SSL_REJECT_UNAUTHORIZED=false,\
N8N_PORT=5678,\
N8N_LISTEN_ADDRESS=0.0.0.0,\
N8N_METRICS=true,\
GCS_BUCKET=${BUCKET_NAME},\
GENERIC_TIMEZONE=America/Argentina/Buenos_Aires" \
    --set-secrets="\
DB_POSTGRESDB_PASSWORD=neon-db-password:latest,\
N8N_ENCRYPTION_KEY=n8n-encryption-key:latest,\
FAL_API_KEY=fal-api-key:latest,\
FAL_KEY=fal-api-key:latest,\
OPENROUTER_API_KEY=openrouter-api-key:latest" \
    --project="$PROJECT_ID"

# ── Paso 2: Obtener URL y actualizar webhook ─────────────────────────────────
echo ""
echo "📦 [2/2] Obteniendo URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format="value(status.url)")

gcloud run services update "$SERVICE_NAME" \
    --region="$REGION" \
    --update-env-vars="WEBHOOK_URL=${SERVICE_URL},N8N_EDITOR_BASE_URL=${SERVICE_URL}" \
    --project="$PROJECT_ID"

echo ""
echo "✅ ¡Despliegue completado!"
echo ""
echo "   🌐 URL: $SERVICE_URL"
echo "   📊 Logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo ""
echo "💡 Escala a CERO → \$0 cuando no hay tráfico."
