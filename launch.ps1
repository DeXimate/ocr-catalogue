$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = 'http://127.0.0.1:8765/'
$bundledPython = 'C:\Users\m.harbaoui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }

$alreadyRunning = $false
try {
    $alreadyRunning = (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1).StatusCode -eq 200
} catch {}

if (-not $alreadyRunning) {
    Start-Process -WindowStyle Hidden -FilePath $python -ArgumentList '-m','ocr_catalogue.app' -WorkingDirectory $project
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 350
        try {
            if ((Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1).StatusCode -eq 200) { break }
        } catch {}
    }
}

Start-Process $url
