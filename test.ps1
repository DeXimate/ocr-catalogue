$ErrorActionPreference = 'Stop'
$bundledPython = 'C:\Users\m.harbaoui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }
& $python -m unittest discover -s tests -v

