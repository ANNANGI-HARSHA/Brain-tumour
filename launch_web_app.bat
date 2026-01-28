@echo off
echo ============================================
echo Brain Tumor Density Analyzer - Web App
echo ============================================
echo.
echo Starting web interface...
echo.
echo The application will open in your browser automatically.
echo If not, open: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================
echo.

cd /d "%~dp0app"
call ..\..venv\Scripts\activate
streamlit run tumor_analyzer_app.py

pause
