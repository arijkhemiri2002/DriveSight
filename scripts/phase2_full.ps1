<#
DriveSight Phase 2 - full run on all 1,161 clean sequences, then packs the evidence.

    powershell -ExecutionPolicy Bypass -File scripts\phase2_full.ps1

  1. test suite
  2. gaze crop diagnostic (letterbox-bar signal; read-only)
  3. full preprocessing run (~20-25 min; resumable: rerun this script after an interruption)
  4. independent verification of all outputs
  5. zips reports, logs, merged CSVs, per-sequence stats and QC images to the Desktop
#>
param([string]$Python = "python")
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "src\features\gaze_bar_diagnostic.py")) { throw "Extract drivesight_phase2_update.zip into the repo first." }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$console = "reports\video_dataset\phase2_full_console_$ts.txt"
function Step($name, [scriptblock]$cmd) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    "`n== $name ==" | Out-File -Append -Encoding utf8 $console
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmd 2>&1 | Tee-Object -Append -FilePath $console | Out-Host
    $code = $LASTEXITCODE
    "[$name] exit code $code, $([math]::Round($sw.Elapsed.TotalMinutes,1)) min" | Tee-Object -Append -FilePath $console | Out-Host
    return $code
}

$t = Step "1. Tests" { & $Python -m pytest -q tests }
if ($t -ne 0) { throw "Tests failed: send $console to Claude." }
Step "2. Gaze crop diagnostic" { & $Python -m src.features.gaze_bar_diagnostic } | Out-Null
Step "3. Full run" { & $Python -m src.features.run_preprocessing } | Out-Null
Step "4. Full verification" { & $Python -m src.features.verify_preprocessing } | Out-Null

Write-Host "`n== 5. Packing results ==" -ForegroundColor Cyan
$stage = Join-Path $env:TEMP "drivesight_full_$ts"
New-Item -ItemType Directory -Force $stage | Out-Null
$items = @("reports\video_dataset\gaze_bar_diagnostic.md", "reports\video_dataset\gaze_bar_diagnostic.png",
           "reports\video_dataset\gaze_bar_diagnostic.csv", "reports\video_dataset\phase2_verification_full.md",
           "reports\video_dataset\gps_validity_scan.md", $console)
$items += Get-ChildItem "data\processed\*\metadata\aligned_records.csv", "data\processed\*\metadata\sequence_stats.csv",
    "data\processed\*\metadata\_schema.json", "data\processed\*\logs\processing_log.csv",
    "data\processed\*\qc\overlays\*.jpg" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
$items += Get-ChildItem "data\processed\logs\*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 2 | ForEach-Object { $_.FullName }
$root = (Get-Location).Path
foreach ($i in $items) {
    if (Test-Path $i) {
        $full = (Resolve-Path $i).Path
        $dst = Join-Path $stage $full.Substring($root.Length + 1)
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Copy-Item $full $dst
    }
}
$zip = Join-Path ([Environment]::GetFolderPath("Desktop")) "drivesight_phase2_full_results.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item -Recurse -Force $stage
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "`nAttach this file in the chat ($mb MB): $zip" -ForegroundColor Green
