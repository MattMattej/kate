#!/bin/bash
# =============================================================================
# Pixel_Kate — Service Account + GCS Bucket Setup
# Crea el Service Account para n8n con los permisos mínimos necesarios
# Ejecutar ANTES de secrets_setup.sh y deploy.sh
# =============================================================================

set -e

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-TU_PROJECT_ID}"
REGION="${REGION:-us-central1}"
SA_NAME="n8n-sa"
SA_DISPLAY_NAME="Pixel_Kate n8n Service Account"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME="pixelkate-media-${PROJECT_ID}"

if [ "$PROJECT_ID" = "TU_PROJECT_ID" ]; then
    echo "❌ ERROR: Define PROJECT_ID antes de ejecutar"
    exit 1
fi

echo "🔑 Configurando Service Account e IAM para Pixel_Kate..."

# ── Paso 1: Crear Service Account ─────────────────────────────────────────────
echo ""
echo "📦 [1/4] Creando Service Account '$SA_NAME'..."
gcloud iam service-accounts create "$SA_NAME" \
    --display-name="$SA_DISPLAY_NAME" \
    --project="$PROJECT_ID" || echo "   ℹ️ Service Account ya existe, continuando..."

# ── Paso 2: Crear bucket de Cloud Storage ──────────────────────────────────────
echo ""
echo "📦 [2/4] Creando bucket de Cloud Storage..."
gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access 2>/dev/null || \
    echo "   ℹ️ Bucket ya existe, continuando..."

# Dar acceso al SA al bucket
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.objectAdmin"

echo "   ✅ Bucket: gs://$BUCKET_NAME"

# ── Paso 3: Asignar rol de Secret Manager ──────────────────────────────────────
echo ""
echo "📦 [3/4] Asignando rol Secret Manager Accessor..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

# ── Paso 4: Asignar rol de Cloud Run Invoker ──────────────────────────────────
echo ""
echo "📦 [4/4] Asignando rol Cloud Run Invoker..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/run.invoker" \
    --quiet

echo ""
echo "✅ Service Account configurado correctamente."
echo ""
echo "   SA Email:    $SA_EMAIL"
echo "   Bucket:      gs://$BUCKET_NAME"
echo ""
echo "📋 Roles asignados:"
gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --format="table(bindings.role, bindings.members)" \
    --filter="bindings.members:$SA_EMAIL"

echo ""
echo "📤 Variables para deploy.sh:"
echo "   export SA_EMAIL=$SA_EMAIL"
echo "   export BUCKET_NAME=$BUCKET_NAME"
