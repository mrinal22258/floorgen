@echo off
REM ============================================================
REM FloorGen: Master One-Click Runner (Windows)
REM ============================================================
title FloorGen SOTA Architecture Engine

if "%~1"=="" (
    echo ================================================================================
    echo               F L O O R G E N   ---   M A S T E R   R U N N E R
    echo ================================================================================
    echo   [1] Launch Interactive Web Studio (Browser) -- DEFAULT
    echo   [2] Run Full Setup, Verification and Synthesis
    echo   [3] Launch FastAPI REST Backend Server (Swagger docs at :8000/docs)
    echo   [4] Run Comprehensive Test Suite
    echo ================================================================================
    set /p choice="Select option [1-4] (Press Enter for 1 - Web Studio): "
    if "%choice%"=="2" (
        python "%~dp0run.py" --all
    ) else if "%choice%"=="3" (
        python "%~dp0run.py" --api
    ) else if "%choice%"=="4" (
        python "%~dp0run.py" --test
    ) else (
        python "%~dp0run.py" --demo
    )
) else (
    python "%~dp0run.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FloorGen] Process finished with exit code %ERRORLEVEL%.
    pause
)
