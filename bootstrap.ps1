# Create .venv next to this script and install pinned requirements.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "py"
    $PyArgs = @("-3.12")
} else {
    $PyArgs = @()
}
$Venv = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $Py @PyArgs -m venv $Venv
}
$VenvPy = Join-Path $Venv "Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")
Write-Host "QuixelStylizer venv ready: $VenvPy"
