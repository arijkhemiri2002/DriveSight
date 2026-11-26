# DriveSight Phase 4 - tracking on the stored Phase 3 detections (CPU only, no video, no YOLO).
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\phase4_run.ps1
# Uses the FROZEN tracker setting (defaults of src\tracking\track_config.py, chosen on validation).
# Add -Retune to redo the validation tuning first (about 20 min). Stops at the first failing step.
param([switch]$Retune)
$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title" -ForegroundColor Red; exit $LASTEXITCODE }
}

Step "tests"                        { & $py -m pytest -q tests }
Step "Phase 2 intact"               { & $py -m src.features.verify_preprocessing }
Step "Phase 3 intact"               { & $py -m src.detection.verify_detection }
if ($Retune) {
    Step "tuning on validation"     { & $py -m src.tracking.tune_tracking }
    Write-Host "Read reports\video_dataset\phase4_tuning_validation.md. If the recommendation differs from the frozen" -ForegroundColor Yellow
    Write-Host "setting, update track_config.py and docs\phase4_tracking.md (section 8) with the reason, then rerun." -ForegroundColor Yellow
}
Step "pilot tracking"               { & $py -m src.tracking.run_tracking --pilot }
Step "pilot verification"           { & $py -m src.tracking.verify_tracking --pilot }
Step "full tracking (all splits)"   { & $py -m src.tracking.run_tracking }
Step "full verification"            { & $py -m src.tracking.verify_tracking }
Step "evaluation (validation)"      { & $py -m src.tracking.evaluate_tracking --split validation }
Step "evaluation (training)"        { & $py -m src.tracking.evaluate_tracking --split training }
Write-Host "`nPhase 4 done. Visual check (needs the camera videos):  $py -m src.tracking.draw_tracks --split validation --n 5" -ForegroundColor Green
Write-Host "Test-split evaluation: once, at the end of the project, with the frozen setting:  $py -m src.tracking.evaluate_tracking --split test"
