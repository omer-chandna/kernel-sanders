$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "py -3.12" }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $py -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pytest pandas matplotlib pyyaml ninja cmake

if (-not (Test-Path .conda-cuda\Library\bin\nvcc.exe)) {
    conda create -y -p "$Root\.conda-cuda" -c nvidia `
        cuda-nvcc=12.8.93 cuda-cudart-dev=12.8.90 libcublas-dev=12.8.4.1 `
        cuda-cccl=12.8.90 cuda-crt=12.8.93
}

Write-Output "Environment ready. Next: powershell -File scripts\build.ps1"
Write-Output "This script does not install a display driver and does not require the CUDA 13.5 system installer."
