#!/usr/bin/env pwsh
# ============================================================
# build_and_push.ps1  —  Pixel Kate RunPod Worker v6
# ============================================================
# Uso desde pixel_kate/:
#   .\runpod_worker\build_and_push.ps1 -DockerUser TU_USUARIO
#
# O con tag personalizado:
#   .\runpod_worker\build_and_push.ps1 -DockerUser TU_USUARIO -Tag v6
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerUser,

    [string]$ImageName = "pixel-kate-worker",
    [string]$Tag = "latest"
)

$IMAGE = "$DockerUser/${ImageName}:${Tag}"
$CONTEXT = "$PSScriptRoot"  # carpeta runpod_worker/

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " Pixel Kate RunPod Worker — Build & Push" -ForegroundColor Cyan
Write-Host "  Imagen: $IMAGE" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Build
Write-Host "📦 Construyendo imagen Docker..." -ForegroundColor Yellow
docker build -t $IMAGE $CONTEXT
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ docker build falló (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build OK" -ForegroundColor Green

# 2. Push
Write-Host ""
Write-Host "🚀 Subiendo al Docker Hub..." -ForegroundColor Yellow
docker push $IMAGE
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ docker push falló (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ LISTO: $IMAGE publicada" -ForegroundColor Green
Write-Host ""
Write-Host "Próximos pasos en RunPod:" -ForegroundColor Cyan
Write-Host "  1. Ir a Serverless → tu endpoint → Edit" -ForegroundColor White
Write-Host "  2. Docker Image: $IMAGE" -ForegroundColor White
Write-Host "  3. Environment Variables:" -ForegroundColor White
Write-Host "       CPU_OFFLOAD_TYPE = sequential" -ForegroundColor White
Write-Host "       MAX_IMAGE_SIDE   = 768" -ForegroundColor White
Write-Host "       DEFAULT_STEPS    = 20" -ForegroundColor White
Write-Host "       LORA1_SCALE      = 0.85" -ForegroundColor White
Write-Host "       LORA2_SCALE      = 0.5" -ForegroundColor White
Write-Host "       HF_TOKEN         = (tu token si necesitas modelo privado)" -ForegroundColor White
Write-Host "  4. Max Workers: 1 (evita OOM con múltiples instancias)" -ForegroundColor White
Write-Host "  5. Save → Restart workers" -ForegroundColor White
Write-Host ""
