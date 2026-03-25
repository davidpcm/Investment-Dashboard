@echo off
title Investment Portfolio Dashboard
echo ============================================
echo   Investment Portfolio Dashboard
echo ============================================
echo.
echo Starting dashboard at http://localhost:8501
echo Press Ctrl+C to stop.
echo.

C:\Users\DAVIDCHMP\Kiro-Project\projects\investment-dashboard\.venv\Scripts\streamlit.exe run C:\Users\DAVIDCHMP\Kiro-Project\projects\investment-dashboard\app.py --server.headless true --browser.gatherUsageStats false

pause
