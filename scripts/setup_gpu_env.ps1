<#
DriveSight - create a project virtual environment (.venv) with a CUDA build of PyTorch.

    powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_env.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_env.ps1 -Cuda cu130     # if cu126 fails

Why a separate environment: the global Python has a CPU-only torch 2.14.1 that other projects
(sentence-transformers, ...) use. The .venv keeps DriveSight's GPU stack isolated and its exact
versions recordable. It is Git-ignored (.venv/). Download size: about 2.5-3 GB.

GPU: NVIDIA GeForce MX550 (Turing, 2 GB), driver CUDA 13.3. CUDA 12.6 builds are used by default:
they support Turing and need nothing installed besides the NVIDIA driver.
#>
param([string]$Cuda = "cu126", [string]$BasePython = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    & $BasePython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}
& $py -m pip install --upgrade pip --quiet

Write-Host "== torch with CUDA ($Cuda) ==" -ForegroundColor Cyan
& $py -m pip install "torch==2.14.1" "torchvision==0.29.2" --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "torch 2.14.1 not found for $Cuda; installing the latest build for $Cuda instead"
    & $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { throw "torch install failed: try -Cuda cu130" }
}

Write-Host "== project packages (versions of the environment used so far) ==" -ForegroundColor Cyan
& $py -m pip install "ultralytics==8.4.189" "opencv-python==5.0.0.93" "numpy==2.4.6" "pandas==2.2.3" `
    "pyarrow==26.0.0" "matplotlib==3.10.3" "scipy==1.15.3" "tabulate==0.10.0" "psutil" "pytest>=8"
if ($LASTEXITCODE -ne 0) { throw "package install failed" }

Write-Host "== check ==" -ForegroundColor Cyan
& $py -c "import torch, ultralytics; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0)); x = torch.ones(2048, 2048, device='cuda', dtype=torch.half); print('matmul ok', float((x @ x)[0, 0])); print('ultralytics', ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) { throw "CUDA check failed: send this output to Claude" }
& $py -m pip freeze | Out-File -Encoding utf8 "reports\video_dataset\venv_pip_freeze.txt"
Write-Host "`nReady. Next: powershell -ExecutionPolicy Bypass -File scripts\phase3_run.ps1   (it uses .venv automatically)" -ForegroundColor Green
