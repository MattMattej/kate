#!/bin/bash
# =============================================================================
# Pixel_Kate — Cloud SQL + pgvector Setup
# Ejecutar en: Cloud Shell o terminal con gcloud autenticado
# Prerequisito: gcloud auth login && gcloud config set project $PROJECT_ID
# =============================================================================

set -e  # Salir si hay error

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
# ⚠️ EDITAR ESTAS VARIABLES ANTES DE EJECUTAR
PROJECT_ID="${PROJECT_ID:-TU_PROJECT_ID}"
REGION="${REGION:-us-central1}"
INSTANCE_NAME="pixelkate-db"
DB_NAME="pixelkate"
DB_USER="n8n-user"
DB_ROOT_PASSWORD="${DB_ROOT_PASSWORD:-}"  # Se pedirá si está vacío
DB_USER_PASSWORD="${DB_USER_PASSWORD:-}"  # Se pedirá si está vacío

# ── Validaciones ──────────────────────────────────────────────────────────────
if [ "$PROJECT_ID" = "TU_PROJECT_ID" ]; then
    echo "❌ ERROR: Define PROJECT_ID antes de ejecutar"
    echo "   export PROJECT_ID=mi-proyecto-gcp"
    exit 1
fi

if [ -z "$DB_ROOT_PASSWORD" ]; then
    read -s -p "🔐 Ingresa contraseña ROOT para Cloud SQL: " DB_ROOT_PASSWORD
    echo ""
fi

if [ -z "$DB_USER_PASSWORD" ]; then
    read -s -p "🔐 Ingresa contraseña para el usuario n8n: " DB_USER_PASSWORD
    echo ""
fi

echo ""
echo "🚀 Configurando Cloud SQL para Pixel_Kate..."
echo "   Proyecto: $PROJECT_ID"
echo "   Región:   $REGION"
echo "   Instancia: $INSTANCE_NAME"

# ── Paso 1: Crear instancia Cloud SQL (PostgreSQL 15) ─────────────────────────
echo ""
echo "📦 [1/5] Creando instancia Cloud SQL..."
gcloud sql instances create "$INSTANCE_NAME" \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region="$REGION" \
    --root-password="$DB_ROOT_PASSWORD" \
    --no-backup \
    --project="$PROJECT_ID"

echo "✅ Instancia creada. Esperando que esté RUNNABLE..."
gcloud sql instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --format="value(state)"

# ── Paso 2: Crear base de datos ───────────────────────────────────────────────
echo ""
echo "📦 [2/5] Creando base de datos '$DB_NAME'..."
gcloud sql databases create "$DB_NAME" \
    --instance="$INSTANCE_NAME" \
    --project="$PROJECT_ID"

# ── Paso 3: Crear usuario n8n ─────────────────────────────────────────────────
echo ""
echo "📦 [3/5] Creando usuario '$DB_USER'..."
gcloud sql users create "$DB_USER" \
    --instance="$INSTANCE_NAME" \
    --password="$DB_USER_PASSWORD" \
    --project="$PROJECT_ID"

# ── Paso 4: Habilitar extensión pgvector (via proxy o Cloud Shell SQL) ────────
echo ""
echo "📦 [4/5] Instrucciones para habilitar pgvector:"
echo "   Conéctate a la DB y ejecuta:"
echo "   gcloud sql connect $INSTANCE_NAME --user=postgres --project=$PROJECT_ID"
echo "   Luego en psql: CREATE EXTENSION vector;"

# ── Paso 5: Exportar variables ────────────────────────────────────────────────
echo ""
echo "📦 [5/5] Variables de entorno para los siguientes scripts:"
INSTANCE_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"
echo "   INSTANCE_CONNECTION_NAME=$INSTANCE_CONNECTION_NAME"
echo ""
echo "✅ Cloud SQL configurado correctamente."
echo "   Estado: $(gcloud sql instances describe $INSTANCE_NAME --project=$PROJECT_ID --format='value(state)')"
