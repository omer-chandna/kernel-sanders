$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$cudaCandidates = @(
    (Join-Path $Root ".conda-cuda\Library"),
    $env:CUDA_PATH,
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin\nvcc.exe")) }
if (-not $cudaCandidates) {
    throw "CUDA toolkit not found. Install CUDA 12.8+ or 13.x (nvcc 11.2 cannot target sm_89)."
}
$Cuda = $cudaCandidates[0]
$env:CUDA_PATH = $Cuda
$env:PATH = "$Cuda\bin;" + $env:PATH

$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "VS 2022 Build Tools vcvars64.bat not found."
}
$cmake = Join-Path $Root ".venv\Scripts\cmake.exe"
$ninja = Join-Path $Root ".venv\Scripts\ninja.exe"
if (-not (Test-Path $cmake)) { $cmake = "cmake" }

$cmd = "`"$vcvars`" && `"$cmake`" -S `"$Root`" -B `"$Root\build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=`"$Cuda\bin\nvcc.exe`" -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_MAKE_PROGRAM=`"$ninja`" && `"$cmake`" --build `"$Root\build`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Built $Root\build\adakernel.dll with $Cuda"
