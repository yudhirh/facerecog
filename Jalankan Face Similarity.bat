@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 goto use_py

where python >nul 2>nul
if errorlevel 1 goto missing_python
set "PYTHON=python"
goto install_dependencies

:use_py
set "PYTHON=py"

:install_dependencies
echo Memasang atau memeriksa kebutuhan aplikasi. Tunggu sebentar...
%PYTHON% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto install_failed

echo Membuka Face Similarity Checker...
%PYTHON% -m streamlit run "%~dp0app.py"
if errorlevel 1 echo Aplikasi berhenti karena terjadi kesalahan.
pause
exit /b

:missing_python
echo Python belum ditemukan. Install Python terlebih dahulu dari https://www.python.org/downloads/
echo Saat instalasi, centang pilihan untuk menambahkan Python ke PATH.
pause
exit /b 1

:install_failed
echo Gagal memasang kebutuhan aplikasi. Periksa koneksi internet dan pesan error di atas.
pause
exit /b 1