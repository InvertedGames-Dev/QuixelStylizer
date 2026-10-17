@echo off
setlocal
title QuixelStylizer
rem Double-click -> GUI. Drag folders onto me -> qs stylize.
set "ROOT=%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYEXE="
set "PYARG="
if exist "%ROOT%.venv\Scripts\python.exe" set "PYEXE=%ROOT%.venv\Scripts\python.exe"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"
if not defined PYEXE (
  where py >nul 2>nul && (set "PYEXE=py" & set "PYARG=-3.12")
)
if not defined PYEXE (
  echo Python 3.12 not found. Install it with:
  echo   winget install -e --id Python.Python.3.12
  echo then run bootstrap.ps1
  pause
  exit /b 1
)
if "%~1"=="" (
  "%PYEXE%" %PYARG% -B "%ROOT%run.py" --gui
  if errorlevel 1 pause
) else (
  echo Stylizing: %*
  echo.
  "%PYEXE%" %PYARG% -B "%ROOT%run.py" stylize %*
  echo.
  echo Log file: "%ROOT%stylizer.log"
  pause
)
endlocal
