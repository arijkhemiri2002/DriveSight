<#
DriveSight Phase 3 - repair the degenerate box(es), re-verify, pack the evidence (no parquet).

    powershell -ExecutionPolicy Bypass -File scripts\phase3_finish.ps1
#>
param([string]$Python = "")
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $Python) { $Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" } }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$console = "reports\video_dataset\phase3_finish_console_$ts.txt"
function Step($name, [scriptblock]$cmd) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    "`n== $name ==" | Out-File -Append -Encoding utf8 $console
    & $cmd 2>&1 | Tee-Object -Append -FilePath $console | Out-Host
    "[$name] exit code $LASTEXITCODE" | Tee-Object -Append -FilePath $console | Out-Host
}
Step "1. Tests" { & $Python -m pytest -q tests }
Step "2. Repair (dry run)" { & $Python -m src.detection.repair_degenerate_boxes --dry-run }
Step "3. Repair" { & $Python -m src.detection.repair_degenerate_boxes }
Step "4. Full verification" { & $Python -m src.detection.verify_detection }
Step "5. Device used" { & $Python -c "import json,glob; [print(p, json.load(open(p))['env']) for p in glob.glob('data/processed/*/detections/_schema.json')]" }

$stage = Join-Path $env:TEMP "drivesight_p3f_$ts"
New-Item -ItemType Directory -Force $stage | Out-Null
$items = @("reports\video_dataset\phase3_env_benchmark.md", "reports\video_dataset\phase3_env_benchmark.json",
           "reports\video_dataset\phase3_verification_full.md", "reports\video_dataset\phase3_repair_degenerate_boxes.md",
           "reports\video_dataset\venv_pip_freeze.txt", $console)
$items += Get-ChildItem "data\processed\*\detections\frame_summary.csv", "data\processed\*\detections\_schema.json",
    "data\processed\*\qc\detections\*.jpg", "reports\video_dataset\phase3_console_*.txt" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
$root = (Get-Location).Path
foreach ($i in $items) {
    if (Test-Path $i) {
        $full = (Resolve-Path $i).Path
        $dst = Join-Path $stage $full.Substring($root.Length + 1)
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Copy-Item $full $dst
    }
}
$zip = Join-Path ([Environment]::GetFolderPath("Desktop")) "drivesight_phase3_final.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item -Recurse -Force $stage
Write-Host "`nAttach this file in the chat ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB): $zip" -ForegroundColor Green
