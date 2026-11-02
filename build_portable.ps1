$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$sitePackages = Join-Path $projectRoot ".venv\Lib\site-packages"
$outputDirectory = Join-Path $projectRoot "dist\FaceSimilarityPortable"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python project tidak ditemukan: $pythonExe. Siapkan .venv terlebih dahulu."
}
if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
    throw "Folder dependency project tidak ditemukan: $sitePackages."
}
if (Test-Path -LiteralPath $outputDirectory) {
    throw "Folder output sudah ada: $outputDirectory. Pindahkan/rename folder itu sebelum build ulang."
}

$pythonVersion = (& $pythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
$pythonBits = (& $pythonExe -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($pythonBits -ne "64") {
    throw "Build portable ini memerlukan Python 64-bit; interpreter saat ini $pythonBits-bit."
}

$weightSource = Join-Path $env:USERPROFILE ".deepface\weights"
$requiredWeights = @("arcface_weights.h5", "retinaface.h5")
foreach ($weight in $requiredWeights) {
    if (-not (Test-Path -LiteralPath (Join-Path $weightSource $weight) -PathType Leaf)) {
        throw "Model $weight belum tersedia di $weightSource. Jalankan aplikasi dengan internet sekali untuk mengunduh model, lalu build ulang."
    }
}

$runtimeUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip"
$runtimeDirectory = Join-Path $outputDirectory "runtime"
$runtimeZip = Join-Path ([IO.Path]::GetTempPath()) ("python-embed-" + [guid]::NewGuid().ToString("N") + ".zip")

try {
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    Write-Host "Mengunduh Python $pythonVersion embeddable..."
    Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip -UseBasicParsing
    Expand-Archive -LiteralPath $runtimeZip -DestinationPath $runtimeDirectory

    $runtimePth = Get-ChildItem -LiteralPath $runtimeDirectory -Filter "python*._pth" |
        Select-Object -First 1
    if ($null -eq $runtimePth) {
        throw "File konfigurasi _pth tidak ditemukan di runtime Python."
    }

    Write-Host "Menyalin dependency Python (sekitar 2 GB)..."
    $runtimeSitePackages = Join-Path $runtimeDirectory "Lib\site-packages"
    New-Item -ItemType Directory -Path $runtimeSitePackages -Force | Out-Null
    & robocopy.exe $sitePackages $runtimeSitePackages /E /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "Gagal menyalin dependency Python (robocopy exit code $LASTEXITCODE)."
    }

    $pthLines = @(Get-Content -LiteralPath $runtimePth.FullName |
        Where-Object {
            $line = $_.Trim()
            $line -and -not $line.StartsWith("#") -and $line -ne "import site"
        })
    if ($pthLines -notcontains "Lib\site-packages") {
        $pthLines += "Lib\site-packages"
    }
    $pthLines += "import site"
    Set-Content -LiteralPath $runtimePth.FullName -Value $pthLines -Encoding Ascii

    Copy-Item -LiteralPath (Join-Path $projectRoot "app.py") -Destination $outputDirectory
    Copy-Item -LiteralPath (Join-Path $projectRoot "Jalankan Face Similarity Portable.bat") -Destination (Join-Path $outputDirectory "Jalankan Face Similarity.bat")
    Copy-Item -LiteralPath (Join-Path $projectRoot "README_PORTABLE.txt") -Destination $outputDirectory

    $portableWeights = Join-Path $outputDirectory "data\.deepface\weights"
    New-Item -ItemType Directory -Path $portableWeights -Force | Out-Null
    foreach ($weight in $requiredWeights) {
        Copy-Item -LiteralPath (Join-Path $weightSource $weight) -Destination $portableWeights
    }

    $cascade = Join-Path $runtimeSitePackages "cv2\data\haarcascade_frontalface_default.xml"
    if (-not (Test-Path -LiteralPath $cascade -PathType Leaf)) {
        throw "OpenCV Haar cascade tidak ditemukan di dependency yang disalin: $cascade"
    }

    Write-Host "Memverifikasi runtime portable..."
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:DEEPFACE_HOME = Join-Path $outputDirectory "data"
    $env:FACE_SIMILARITY_OFFLINE = "1"
    $env:TF_CPP_MIN_LOG_LEVEL = "2"
    $runtimePython = Join-Path $runtimeDirectory "python.exe"
    & $runtimePython -c "import cv2, deepface, numpy, streamlit, tensorflow, tf_keras; print('Runtime dan dependency utama siap.')"
    if ($LASTEXITCODE -ne 0) {
        throw "Verifikasi runtime gagal. Lihat pesan error di atas."
    }

    Write-Host ""
    Write-Host "Paket portable berhasil dibuat:"
    Write-Host $outputDirectory
    Write-Host "Salin seluruh folder tersebut ke PC Windows x64, lalu jalankan Jalankan Face Similarity.bat."
}
catch {
    Write-Error $_
    throw
}
finally {
    if (Test-Path -LiteralPath $runtimeZip) {
        Remove-Item -LiteralPath $runtimeZip -Force
    }
}
