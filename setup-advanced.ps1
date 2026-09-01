$ErrorActionPreference = 'Stop'
$bundledPython = 'C:\Users\m.harbaoui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }
Write-Host 'Installation des moteurs OCR et de mise en page locaux...'
& $python -m pip install -r requirements-advanced.txt
Write-Host 'Installation terminée. Les poids des modèles seront téléchargés à leur première utilisation.'
