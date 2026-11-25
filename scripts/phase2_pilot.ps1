<#
DriveSight Phase 2 - Steps 1, 4, 5 on real data, then packs the evidence into one zip.

    powershell -ExecutionPolicy Bypass -File scripts\phase2_pilot.ps1

  1. GPS validity scan of every BDDA GPS file (read-only)
  2. frame accuracy check: pilot-style seeking vs sequential decoding (read-only)
  3. extended pilot run (3 per split + 1744 + a 60 fps clip + the longest clip),
     with a QC overlay for every pilot sequence
  4. independent verification of the pilot outputs
  5. zips reports, logs, per-sequence CSVs and QC images to the Desktop

Steps that fail do not stop the others; every result goes into the zip.
#>
param([string]$Python = "python")
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "src\features\run_preprocessing.py")) { throw "Run step0_apply_phase2.ps1 first." }
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$console = "reports\video_dataset\phase2_pilot_console_$ts.txt"
function Step($name, [scriptblock]$cmd) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    "`n== $name ==" | Out-File -Append -Encoding utf8 $console
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmd 2>&1 | Tee-Object -Append -FilePath $console
    "[$name] exit code $LASTEXITCODE, $([math]::Round($sw.Elapsed.TotalSeconds,1)) s" | Tee-Object -Append -FilePath $console
}

Step "1. GPS validity scan"        { & $Python -m src.features.gps_validity_scan }
Step "2. Frame accuracy check"     { & $Python -m src.features.check_frame_accuracy }
Step "3. Pilot run"                { & $Python -m src.features.run_preprocessing --pilot --qc-fraction 1.0 }
Step "4. Pilot verification"       { & $Python -m src.features.verify_preprocessing --pilot }

Write-Host "`n== 5. Packing results ==" -ForegroundColor Cyan
$stage = Join-Path $env:TEMP "drivesight_pilot_$ts"
New-Item -ItemType Directory -Force $stage | Out-Null
$items = @(
    "reports\video_dataset\gps_validity_scan.md", "reports\video_dataset\gps_validity_scan.csv",
    "reports\video_dataset\frame_accuracy_check.md", "reports\video_dataset\phase2_verification_pilot.md",
    $console
)
$items += Get-ChildItem "data\processed\*\metadata\sequence_stats.csv", "data\processed\*\metadata\_schema.json",
    "data\processed\*\logs\processing_log.csv", "data\processed\*\qc\overlays\*.jpg",
    "data\processed\*\metadata\sequence_records\*.csv" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
$items += Get-ChildItem "data\processed\logs\*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 2 | ForEach-Object { $_.FullName }
$root = (Get-Location).Path
foreach ($i in $items) {
    if (Test-Path $i) {
        $full = (Resolve-Path $i).Path
        $rel = $full.Substring($root.Length + 1)
        $dst = Join-Path $stage $rel
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Copy-Item $full $dst
    }
}
$zip = Join-Path ([Environment]::GetFolderPath("Desktop")) "drivesight_phase2_pilot_results.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip
Remove-Item -Recurse -Force $stage
Write-Host "`nAttach this file in the chat: $zip" -ForegroundColor Green
