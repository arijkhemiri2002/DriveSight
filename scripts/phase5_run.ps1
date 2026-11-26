# DriveSight Phase 5: fusion, verification and risk model (validation only).
# Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File scripts\phase5_run.ps1
# Needs the verified Phase 2 and Phase 4 outputs in data\processed\{split}\metadata and \tracks.
# The test split is NOT scored here: run "python -m src.fusion.train_risk --final-test" once, at the very end.
$ErrorActionPreference = "Stop"

function Step($title, $cmd) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $title" }
}

Step "Phase 5 - fusion (tracks + gaze + GPS, labels)" "python -m src.fusion.run_fusion"
Step "Phase 5 - verification" "python -m src.fusion.verify_fusion"
Step "Phase 5 - risk model (training + validation)" "python -m src.fusion.train_risk"
Step "Tests" "python -m pytest tests -q"

Write-Host ""
Write-Host "Done. Reports: reports\video_dataset\phase5_verification.md and phase5_model_validation.md" -ForegroundColor Green
Write-Host "Models: data\risk\risk_model_fusion.joblib and risk_model_vision_v5.joblib (used by the demo app)." -ForegroundColor Green
