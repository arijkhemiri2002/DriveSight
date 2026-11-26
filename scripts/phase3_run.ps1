<#
DriveSight Phase 3 - YOLOv8s detection on the 62,008 Phase 2 frames, with gates.

    powershell -ExecutionPolicy Bypass -File scripts\phase3_run.ps1            # env check, pilot, full run
    powershell -ExecutionPolicy Bypass -File scripts\phase3_run.ps1 -PilotOnly

  1. tests
  2. environment + benchmark       STOP if torch cannot see the GPU (a CPU run would take ~10 h)
  3. pilot (11 sequences, a QC image for each) + verification     STOP if verification fails
  4. full run (resumable: rerun this script after an interruption) + verification
  5. records the Python environment (pip freeze) and zips the evidence to the Desktop
#>
param([string]$Python = "", [switch]$PilotOnly, [int]$Batch = 0)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "src\detection\run_detection.py")) { throw "Extract drivesight_phase3.zip into the repo first." }
if (-not $Python) { $Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" } }
Write-Host "Python: $Python"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$console = "reports\video_dataset\phase3_console_$ts.txt"
function Step($name, [scriptblock]$cmd) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    "`n== $name ==" | Out-File -Append -Encoding utf8 $console
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmd 2>&1 | Tee-Object -Append -FilePath $console | Out-Host
    $code = $LASTEXITCODE
    "[$name] exit code $code, $([math]::Round($sw.Elapsed.TotalMinutes,1)) min" | Tee-Object -Append -FilePath $console | Out-Host
    return $code
}
function Pack($label) {
    Write-Host "`n== Packing results ==" -ForegroundColor Cyan
    & $Python -m pip freeze 2>$null | Out-File -Encoding utf8 "reports\video_dataset\phase3_environment_pip_freeze.txt"
    $stage = Join-Path $env:TEMP "drivesight_p3_$ts"
    New-Item -ItemType Directory -Force $stage | Out-Null
    $items = @("reports\video_dataset\phase3_env_benchmark.md", "reports\video_dataset\phase3_verification_pilot.md",
               "reports\video_dataset\phase3_verification_full.md", "reports\video_dataset\phase3_environment_pip_freeze.txt", $console)
    $items += Get-ChildItem "data\processed\*\detections\frame_summary.csv", "data\processed\*\detections\_schema.json",
        "data\processed\*\qc\detections\*.jpg" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    $pq = Get-ChildItem "data\processed\*\detections\detections.parquet" -ErrorAction SilentlyContinue
    if ($pq -and (($pq | Measure-Object Length -Sum).Sum -lt 80MB)) { $items += $pq.FullName }
    $items += Get-ChildItem "data\processed\logs\detect_*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 4 | ForEach-Object { $_.FullName }
    $root = (Get-Location).Path
    foreach ($i in $items) {
        if (Test-Path $i) {
            $full = (Resolve-Path $i).Path
            $dst = Join-Path $stage $full.Substring($root.Length + 1)
            New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
            Copy-Item $full $dst
        }
    }
    $zip = Join-Path ([Environment]::GetFolderPath("Desktop")) "drivesight_phase3_$label.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path "$stage\*" -DestinationPath $zip
    Remove-Item -Recurse -Force $stage
    Write-Host "`nAttach this file in the chat ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB): $zip" -ForegroundColor Green
}

if ((Step "1. Tests" { & $Python -m pytest -q tests }) -ne 0) { throw "Tests failed: send $console to Claude." }

Step "2. Environment and benchmark" { & $Python -m src.detection.check_detection_env } | Out-Null
$cuda = & $Python -c "import torch; print(torch.cuda.is_available())"
if ($cuda -ne "True") {
    Write-Warning "torch cannot see the GPU. Install a CUDA build of torch (https://pytorch.org/get-started/locally/), then rerun."
    Pack "env_only"; exit 1
}
$halfArg = @()
# batch size: -Batch if given, else the fastest batch that fitted in GPU memory in the benchmark
if ($Batch -le 0 -and (Test-Path "reports\video_dataset\phase3_env_benchmark.json")) {
    $bj = Get-Content "reports\video_dataset\phase3_env_benchmark.json" | ConvertFrom-Json
    if ($bj.best_batch) { $Batch = [int]$bj.best_batch }
    if ($null -ne $bj.best_half) { $halfArg = @("--half", $(if ($bj.best_half) { "true" } else { "false" })) }
}
$batchArg = @()
if ($Batch -gt 0) { $batchArg = @("--batch", "$Batch"); Write-Host "Batch size: $Batch" }
if ($halfArg) { $batchArg += $halfArg; Write-Host "Precision flag: $($halfArg -join ' ')" }

$c = Step "3a. Pilot detection" { & $Python -m src.detection.run_detection --pilot --qc-fraction 1.0 @batchArg }
$v = Step "3b. Pilot verification" { & $Python -m src.detection.verify_detection --pilot }
if ($v -ne 0 -or $PilotOnly) { Pack "pilot"; if ($v -ne 0) { Write-Warning "Pilot verification failed: stopping." }; exit $v }

Step "4a. Full detection" { & $Python -m src.detection.run_detection @batchArg } | Out-Null
Step "4b. Full verification" { & $Python -m src.detection.verify_detection } | Out-Null
Pack "full"
