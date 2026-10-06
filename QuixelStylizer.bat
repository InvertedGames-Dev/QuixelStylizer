@echo off
setlocal
title QuixelStylizer
rem QuixelStylizer launcher.
rem   Double-click            -> GUI
rem   Drag folder(s) onto me  -> CLI with default settings
set "SCRIPT=%~dp0quixel_stylize.py"
rem no .pyc caching: always run the .py files in this folder
set "PYTHONDONTWRITEBYTECODE=1"
set "PYEXE="
set "PYARG="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"
if not defined PYEXE (
  where py >nul 2>nul && (set "PYEXE=py" & set "PYARG=-3.12")
)
if not defined PYEXE (
  echo Python 3.12 not found. Install it with:
  echo   winget install -e --id Python.Python.3.12
  echo then: python -m pip install numpy opencv-python pillow
  pause
  exit /b 1
)
if "%~1"=="" (
  "%PYEXE%" %PYARG% -B "%SCRIPT%" --gui || pause
) else (
  echo Converting with default settings: %*
  echo.
  "%PYEXE%" %PYARG% -B "%SCRIPT%" %*
  echo.
  echo Log file: "%~dp0stylizer.log"
  pause
)
endlocal
