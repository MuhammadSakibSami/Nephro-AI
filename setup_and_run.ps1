<#
.SYNOPSIS
    One-command setup + run for the Kidney-Risk Screening app (Windows).

.DESCRIPTION
    Creates the Python venv and installs backend deps, prepares .env files,
    installs the frontend, then launches backend (:8000) and frontend (:5173)
    each in its own PowerShell window.

    Run from repo root: folder containing backend/, frontend/, requirements.txt.

.PARAMETER SetupOnly
    Install/prepare everything but do not launch.

.PARAMETER BackendOnly
    Set up + run only the backend.

.PARAMETER FrontendOnly
    Set up + run only the frontend.

.PARAMETER BackendPort
    Backend port. Default 8000.

.PARAMETER FrontendPort
    Frontend dev-server port. Default 5173.

.EXAMPLE
    .\setup_and_run.ps1

.EXAMPLE
    .\setup_and_run.ps1 -SetupOnly

.NOTES
    If PowerShell blocks execution:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>

param(
    [switch]$SetupOnly,
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
Set-Location $Root

function Info($m) { Write-Host ("==> " + $m) -ForegroundColor Cyan }
function Ok($m)   { Write-Host ("    " + $m) -ForegroundColor Green }
function Warn($m) { Write-Host ("    " + $m) -ForegroundColor Yellow }

$doBackend = -not $FrontendOnly
$doFrontend = -not $BackendOnly

# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------
if ($doBackend) {
    Info "Backend: Python virtual environment"

    $venvPython = Join-Path $Root "venv\Scripts\python.exe"

    if (-not (Test-Path $venvPython)) {
        python -m venv venv
        Ok "Created venv"
    }
    else {
        Ok "venv already exists"
    }

    $py = $venvPython

    Info "Backend: installing dependencies from requirements.txt"

    if (-not (Test-Path (Join-Path $Root "requirements.txt"))) {
        throw "requirements.txt not found. Run this script from the repo root."
    }

    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r requirements.txt --quiet
    Ok "Backend dependencies installed"

    $rootEnv = Join-Path $Root ".env"
    $rootEnvExample = Join-Path $Root ".env.example"

    if (-not (Test-Path $rootEnv)) {
        if (Test-Path $rootEnvExample) {
            Copy-Item $rootEnvExample $rootEnv
            Ok "Created .env from .env.example"
        }
        else {
            @"
CKD_LLM_MODEL=local_demo
"@ | Set-Content -Path $rootEnv -Encoding UTF8
            Ok "Created minimal .env with CKD_LLM_MODEL=local_demo"
        }
    }
    else {
        Ok ".env already present"
    }

    $modelDir = Join-Path $Root "model_package"
    $model = Get-ChildItem (Join-Path $modelDir "*.joblib") -ErrorAction SilentlyContinue

    if (-not $model) {
        Warn "No model .joblib found in model_package. The /predict endpoint will not work until you add it."
    }
}

# ---------------------------------------------------------------------------
# Frontend setup
# ---------------------------------------------------------------------------
if ($doFrontend) {
    Info "Frontend: npm install"

    $frontendDir = Join-Path $Root "frontend"

    if (-not (Test-Path $frontendDir)) {
        throw "frontend folder not found. Make sure you run this from the repo root."
    }

    Push-Location $frontendDir

    if (-not (Test-Path "node_modules")) {
        npm install
        Ok "Frontend dependencies installed"
    }
    else {
        Ok "node_modules already present"
    }

    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Ok "Created frontend/.env from .env.example"
        }
        else {
            ("VITE_API_BASE_URL=http://localhost:{0}" -f $BackendPort) | Set-Content -Path ".env" -Encoding UTF8
            Ok "Created minimal frontend/.env"
        }
    }
    else {
        Ok "frontend/.env already present"
    }

    Pop-Location
}

if ($SetupOnly) {
    Info "Setup complete (-SetupOnly). Start later with:"
    Write-Host ("    Backend:  uvicorn backend.main:app --port {0} --reload" -f $BackendPort) -ForegroundColor Gray
    Write-Host "    Frontend: cd frontend; npm run dev" -ForegroundColor Gray
    return
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if ($doBackend) {
    Info ("Launching backend on http://localhost:{0}" -f $BackendPort)

    $activate = Join-Path $Root "venv\Scripts\Activate.ps1"
    $backendCmd = "Set-Location '$Root'; & '$activate'; uvicorn backend.main:app --host 127.0.0.1 --port $BackendPort --reload"

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
    Ok "Backend window started"
}

if ($doFrontend) {
    Info ("Launching frontend on http://localhost:{0}" -f $FrontendPort)

    $frontendDir = Join-Path $Root "frontend"
    $frontendCmd = "Set-Location '$frontendDir'; npm run dev -- --host 127.0.0.1 --port $FrontendPort"

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd
    Ok "Frontend window started"
}

Info "Done."

if ($doFrontend) {
    Write-Host ("    Open http://localhost:{0}" -f $FrontendPort) -ForegroundColor Green
}

if ($doBackend) {
    Write-Host ("    API docs at http://localhost:{0}/docs" -f $BackendPort) -ForegroundColor Green
}