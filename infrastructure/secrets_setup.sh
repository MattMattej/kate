#!/bin/bash
# =============================================================================
# Pixel_Kate — Secret Manager Setup (V1: solo TikTok + Imágenes)
# Ejecutar DESPUÉS de service_account_setup.sh
# =============================================================================

set -e

PROJECT_ID="${PROJECT_ID:-TU_PROJECT_ID}"
SA_EMAIL="n8n-sa@${PROJECT_ID}.iam.gserviceaccount.com"

if [ "$PROJECT_ID" = "TU_PROJECT_ID" ]; then
    echo "❌ ERROR: Define PROJECT_ID"
    exit 1
fi

echo "🔐 Configurando Secret Manager para Pixel_Kate..."

# ── Función helper ────────────────────────────────────────────────────────────
create_secret() {
    local SECRET_NAME="$1"
    local SECRET_PROMPT="$2"
    local SECRET_VALUE="$3"

    if [ -z "$SECRET_VALUE" ]; then
        read -s -p "🔐 Ingresa valor para '$SECRET_NAME' ($SECRET_PROMPT): " SECRET_VALUE
        echo ""
    fi

    if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
        echo "   🔄 Actualizando secreto existente: $SECRET_NAME"
        printf "%s" "$SECRET_VALUE" | \
            gcloud secrets versions add "$SECRET_NAME" \
                --data-file=- \
                --project="$PROJECT_ID"
    else
        echo "   ➕ Creando secreto: $SECRET_NAME"
        printf "%s" "$SECRET_VALUE" | \
            gcloud secrets create "$SECRET_NAME" \
                --data-file=- \
                --replication-policy="automatic" \
                --project="$PROJECT_ID"
    fi

    gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID" \
        --quiet

    echo "   ✅ $SECRET_NAME configurado"
}

# ── 1. Contraseña de Neon ─────────────────────────────────────────────────────
echo ""
echo "📦 [1/2] Contraseña de Neon PostgreSQL..."
create_secret "neon-db-password" "contraseña de Neon (del dashboard)"

# ── 2. Encryption Key ────────────────────────────────────────────────────────
echo ""
echo "📦 [2/2] Clave de encriptación n8n..."
N8N_ENC_KEY=$(openssl rand -base64 32)
create_secret "n8n-encryption-key" "auto-generada" "$N8N_ENC_KEY"
echo "   ⚠️ Guarda esta clave: $N8N_ENC_KEY"

# ── 3. Placeholders para API keys (actualizar cuando tengas las keys) ─────────
echo ""
echo "📦 Creando placeholders para API keys..."

for SECRET_NAME in "fal-api-key" "openrouter-api-key" "tiktok-access-token"; do
    create_secret "$SECRET_NAME" "placeholder" "PLACEHOLDER_UPDATE_LATER"
    echo "   📌 $SECRET_NAME → placeholder"
done

echo ""
echo "✅ Secretos configurados."
echo ""
echo "📋 Secretos creados:"
gcloud secrets list --project="$PROJECT_ID" --format="table(name,createTime)"
echo ""
echo "⚠️  Para actualizar con valores reales:"
echo "   echo -n 'TU_KEY' | gcloud secrets versions add NOMBRE --data-file=- --project=$PROJECT_ID"
