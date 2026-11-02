@echo off
setlocal
cd /d "%~dp0"

set "PYTHONNOUSERSITE=1"
set "PYTHONIOENCODING=utf-8"
set "DEEPFACE_HOME=%~dp0data"
set "FACE_SIMILARITY_OFFLINE=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "TF_CPP_MIN_LOG_LEVEL=2"

if not exist "%~dp0runtime\python.exe" (
    echo Runtime Python portable tidak ditemukan. Pastikan seluruh folder aplikasi disalin.
    pause
    exit /b 1
)

if not exist "%~dp0data\.deepface\weights\arcface_weights.h5" (
    echo Model ArcFace tidak ditemukan. Salin ulang seluruh folder aplikasi.
    pause
    exit /b 1
)

if not exist "%~dp0data\.deepface\weights\retinaface.h5" (
    echo Model RetinaFace tidak ditemukan. Salin ulang seluruh folder aplikasi.
    pause
    exit /b 1
)

echo Membuka Face Similarity Checker dalam mode offline...
"%~dp0runtime\python.exe" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1
if errorlevel 1 (
    echo Aplikasi berhenti karena terjadi kesalahan.
    pause
)
