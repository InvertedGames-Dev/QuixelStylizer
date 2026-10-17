# Rebuild QuixelStylizer.exe from launcher.c (windowed host for run.py).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$ico = Join-Path $Root "quixelstylizer.ico"
$rc = ""
if (Test-Path $ico) {
    Set-Content -Path (Join-Path $Root "launcher.rc") -Value "IDI_ICON1 ICON `"quixelstylizer.ico`""
    $rc = "rc /nologo launcher.rc && "
}
cmd /c "`"$vcvars`" && cd /d `"$Root`" && ${rc}cl /nologo /O2 /MD /I `"$Py\include`" launcher.c /FeQuixelStylizer.exe /link /SUBSYSTEM:WINDOWS /ENTRY:WinMainCRTStartup python312.lib delayimp.lib /DELAYLOAD:python312.dll /LIBPATH:`"$Py\libs`" user32.lib shell32.lib shcore.lib $(if (Test-Path (Join-Path $Root 'launcher.res')) { 'launcher.res' } else { '' })"
Remove-Item -ErrorAction SilentlyContinue (Join-Path $Root "launcher.obj"), (Join-Path $Root "launcher.res"), (Join-Path $Root "launcher.rc")
Write-Host "Wrote $Root\QuixelStylizer.exe"
