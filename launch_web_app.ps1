# Brain Tumor Density Analyzer - Web Application Launcher
# 
# This script launches the Streamlit web interface

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Brain Tumor Density Analyzer - Web App" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting web interface..." -ForegroundColor Yellow
Write-Host ""
Write-Host "The application will open in your browser automatically." -ForegroundColor White
Write-Host "If not, open: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path (Join-Path $PSScriptRoot "app")

# Activate virtual environment
& (Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1")

# Launch Streamlit
streamlit run tumor_analyzer_app.py
