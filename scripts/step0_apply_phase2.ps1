<#
DriveSight Phase 2 - Step 0: housekeeping + install the Phase 2 code.

Run from anywhere (the script moves to the repository root itself):
    powershell -ExecutionPolicy Bypass -File scripts\step0_apply_phase2.ps1 -Package "$HOME\Downloads\drivesight_phase2.zip"

What it does (nothing is deleted; it never commits or pushes):
  1. safety checks: repo root, local src/features unchanged since the audited commit fe00546
  2. git fetch + fast-forward to origin/main (stops if teammates changed src/features)
  3. archives the pilot-v1 outputs (JPEG frames, old CSV, crop QC) to data\processed\_archive_pilot_v1\
  4. stops tracking data/ in Git (git rm --cached: files stay on disk)
  5. moves the 3 superseded scripts to legacy\ (git mv)
  6. fixes .gitignore so new reports in reports\video_dataset\ are not silently ignored
  7. extracts the Phase 2 package over the repo (changed + new files)
  8. installs pytest if missing and runs the test suite
#>
param(
    [Parameter(Mandatory = $true)][string]$Package,
    [string]$Python = "python",
    [string]$AuditedCommit = "fe00546"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$Package = (Resolve-Path $Package).Path
$repo = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repo ".git"))) {
    # the script may be run from the extracted package before it is copied into the repo
    $repo = (Get-Location).Path
}
Set-Location $repo
if (-not (Test-Path ".git") -or -not (Test-Path "src\features")) { throw "Run this from the DriveSight repository root." }
function Invoke-Git { & git.exe @args; if ($LASTEXITCODE -ne 0) { throw "git $args failed" } }

Write-Host "== 1. Safety checks ==" -ForegroundColor Cyan
$dirty = git status --porcelain -- src/features | Where-Object { $_ -notmatch '^\?\? src/features/reports/' }
if ($dirty) { throw "Uncommitted changes in src/features:`n$($dirty -join "`n")`nCommit or stash them first." }

Write-Host "== 2. Sync with origin/main ==" -ForegroundColor Cyan
Invoke-Git fetch origin
$incoming = git log --oneline HEAD..origin/main
if ($incoming) {
    Write-Host "Incoming commits:"; $incoming | ForEach-Object { Write-Host "  $_" }
    git diff --stat HEAD origin/main
    Invoke-Git merge --ff-only origin/main
}
$changed = git diff --name-only $AuditedCommit HEAD -- src/features
if ($changed) {
    throw "src/features changed since the audited commit $AuditedCommit :`n$($changed -join "`n")`nSTOP: send this list to Claude before applying the package."
}

Write-Host "== 3. Archive pilot-v1 outputs (move, not delete) ==" -ForegroundColor Cyan
$arch = "data\processed\_archive_pilot_v1"
foreach ($s in "training", "validation", "test") {
    foreach ($d in "frames", "metadata") {
        $src = "data\processed\$s\$d"
        if (Test-Path $src) {
            New-Item -ItemType Directory -Force "$arch\$s" | Out-Null
            Move-Item $src "$arch\$s\$d"
            Write-Host "  moved $src -> $arch\$s\$d"
        }
    }
}
if (Test-Path "src\features\reports") {
    New-Item -ItemType Directory -Force $arch | Out-Null
    Move-Item "src\features\reports" "$arch\crop_verification_reports"
    Write-Host "  moved src\features\reports -> $arch\crop_verification_reports"
}

Write-Host "== 4. Stop tracking data/ (files stay on disk) ==" -ForegroundColor Cyan
$n = (git ls-files data | Measure-Object).Count
if ($n -gt 0) { Invoke-Git rm -r --cached --quiet data; Write-Host "  untracked $n files under data/" } else { Write-Host "  nothing tracked under data/" }

Write-Host "== 5. Move superseded scripts to legacy\ ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force legacy | Out-Null
$moves = @{
    "src\features\preprocessing.py"       = "legacy\bdd100k_to_yolo_v1_preprocessing.py";
    "src\features\verification.py"        = "legacy\bdd100k_to_yolo_v1_verification.py";
    "src\features\video_preprocessing.py" = "legacy\video_preprocessing_pilot_v1.py"
}
foreach ($k in $moves.Keys) { if (Test-Path $k) { Invoke-Git mv $k $moves[$k]; Write-Host "  $k -> $($moves[$k])" } }

Write-Host "== 6. .gitignore ==" -ForegroundColor Cyan
$gi = Get-Content .gitignore
if ($gi -contains "/reports/") {
    $gi = $gi | ForEach-Object { if ($_ -eq "/reports/") { "/reports/*"; "!/reports/video_dataset/" } else { $_ } }
    if ($gi -notcontains ".pytest_cache/") { $gi += ".pytest_cache/" }
    Set-Content -Path .gitignore -Value $gi -Encoding utf8
    Write-Host "  /reports/ -> /reports/* + !/reports/video_dataset/ ; added .pytest_cache/"
}

Write-Host "== 7. Extract the Phase 2 package ==" -ForegroundColor Cyan
Expand-Archive -Path $Package -DestinationPath . -Force
Get-ChildItem -Recurse -Filter "__pycache__" -Directory src | Remove-Item -Recurse -Force

$old = $null
$nbs = Get-ChildItem notebooks -Recurse -Filter *.ipynb -ErrorAction SilentlyContinue
if ($nbs) {
    $old = Select-String -Path $nbs.FullName -List -ErrorAction SilentlyContinue `
        -Pattern 'extract_frame_from_capture|build_gps_records|nearest_gps|target_timestamps|save_frame\(|frame_alignment_ok|video_preprocessing'
}
if ($old) { Write-Warning "These notebooks use functions that were replaced (they will need updating):"; $old | ForEach-Object { Write-Host "  $($_.Path)" } }

Write-Host "== 8. Tests ==" -ForegroundColor Cyan
& $Python -m pytest --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { & $Python -m pip install --quiet "pytest>=8" }
& $Python -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Tests failed: send the output to Claude." }

Write-Host "`n== Done. Nothing was committed. ==" -ForegroundColor Green
git status --short | Select-Object -First 40
Write-Host "`nNext: powershell -ExecutionPolicy Bypass -File scripts\phase2_pilot.ps1"
