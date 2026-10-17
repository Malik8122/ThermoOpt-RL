# Start the ThermoOpt dashboard: builds the frontend if needed, then serves API + UI on http://127.0.0.1:8000
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path "frontend\node_modules")) { Push-Location frontend; npm install; Pop-Location }
if (-not (Test-Path "frontend\dist\index.html")) { Push-Location frontend; npm run build; Pop-Location }
Start-Process "http://127.0.0.1:8000"
& ".venv\Scripts\python.exe" -m backend.api.main
