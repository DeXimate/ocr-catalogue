$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = 'http://127.0.0.1:8765/'
$bundledPython = 'C:\Users\m.harbaoui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }
$eAcute = [char]0x00E9
$eGrave = [char]0x00E8
$eCircumflex = [char]0x00EA
$aGrave = [char]0x00E0
$apostrophe = [char]0x2019
$longDash = [char]0x2014

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

function Test-OcrServer {
    try {
        return (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1).StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-OcrServer {
    if (-not (Test-OcrServer)) {
        Start-Process -WindowStyle Hidden -FilePath $python -ArgumentList '-m','ocr_catalogue.app' -WorkingDirectory $project
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 300
            if (Test-OcrServer) { break }
        }
    }
    return (Test-OcrServer)
}

function Stop-OcrServer {
    if (-not (Test-OcrServer)) { return $true }
    try {
        Invoke-WebRequest -UseBasicParsing -Method Post -Uri ($url + 'api/shutdown') -ContentType 'application/json' -Body '{}' -TimeoutSec 3 | Out-Null
    } catch {
        return $false
    }
    for ($attempt = 0; $attempt -lt 25; $attempt++) {
        Start-Sleep -Milliseconds 200
        if (-not (Test-OcrServer)) { return $true }
    }
    return $false
}

$form = New-Object System.Windows.Forms.Form
$form.Text = 'OCR Catalogue Monoprix'
$form.ClientSize = New-Object System.Drawing.Size(470, 250)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $true
$form.BackColor = [System.Drawing.Color]::FromArgb(247, 248, 250)
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$brand = New-Object System.Windows.Forms.Panel
$brand.Location = New-Object System.Drawing.Point(0, 0)
$brand.Size = New-Object System.Drawing.Size(470, 6)
$brand.BackColor = [System.Drawing.Color]::FromArgb(220, 20, 55)
$form.Controls.Add($brand)

$title = New-Object System.Windows.Forms.Label
$title.Text = 'OCR Catalogue Monoprix'
$title.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 17)
$title.ForeColor = [System.Drawing.Color]::FromArgb(22, 29, 40)
$title.AutoSize = $true
$title.Location = New-Object System.Drawing.Point(28, 27)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = "Serveur local et acc${eGrave}s ${aGrave} l${apostrophe}application"
$subtitle.ForeColor = [System.Drawing.Color]::FromArgb(104, 113, 128)
$subtitle.AutoSize = $true
$subtitle.Location = New-Object System.Drawing.Point(30, 66)
$form.Controls.Add($subtitle)

$statusDot = New-Object System.Windows.Forms.Panel
$statusDot.Size = New-Object System.Drawing.Size(11, 11)
$statusDot.Location = New-Object System.Drawing.Point(31, 105)
$form.Controls.Add($statusDot)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.AutoSize = $true
$statusLabel.Location = New-Object System.Drawing.Point(50, 100)
$form.Controls.Add($statusLabel)

$openButton = New-Object System.Windows.Forms.Button
$openButton.Text = "Ouvrir l${apostrophe}application"
$openButton.Size = New-Object System.Drawing.Size(190, 44)
$openButton.Location = New-Object System.Drawing.Point(29, 147)
$openButton.FlatStyle = 'Flat'
$openButton.FlatAppearance.BorderSize = 0
$openButton.BackColor = [System.Drawing.Color]::FromArgb(220, 20, 55)
$openButton.ForeColor = [System.Drawing.Color]::White
$openButton.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 10)
$openButton.Cursor = [System.Windows.Forms.Cursors]::Hand
$form.Controls.Add($openButton)

$stopButton = New-Object System.Windows.Forms.Button
$stopButton.Text = "Arr${eCircumflex}ter le serveur"
$stopButton.Size = New-Object System.Drawing.Size(190, 44)
$stopButton.Location = New-Object System.Drawing.Point(235, 147)
$stopButton.FlatStyle = 'Flat'
$stopButton.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(210, 214, 220)
$stopButton.BackColor = [System.Drawing.Color]::White
$stopButton.ForeColor = [System.Drawing.Color]::FromArgb(40, 48, 61)
$stopButton.Font = New-Object System.Drawing.Font('Segoe UI Semibold', 10)
$stopButton.Cursor = [System.Windows.Forms.Cursors]::Hand
$form.Controls.Add($stopButton)

$hint = New-Object System.Windows.Forms.Label
$hint.Text = "Vous pouvez garder cette fen${eCircumflex}tre r${eAcute}duite pendant votre travail."
$hint.ForeColor = [System.Drawing.Color]::FromArgb(120, 128, 140)
$hint.AutoSize = $true
$hint.Location = New-Object System.Drawing.Point(30, 211)
$form.Controls.Add($hint)

function Update-Controls {
    $running = Test-OcrServer
    if ($running) {
        $statusDot.BackColor = [System.Drawing.Color]::FromArgb(25, 158, 108)
        $statusLabel.Text = "Serveur actif ${longDash} http://127.0.0.1:8765"
        $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(22, 120, 84)
        $openButton.Text = "Ouvrir l${apostrophe}application"
        $stopButton.Enabled = $true
    } else {
        $statusDot.BackColor = [System.Drawing.Color]::FromArgb(151, 158, 169)
        $statusLabel.Text = "Serveur arr${eCircumflex}t${eAcute}"
        $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(104, 113, 128)
        $openButton.Text = "D${eAcute}marrer et ouvrir"
        $stopButton.Enabled = $false
    }
}

$openButton.Add_Click({
    $openButton.Enabled = $false
    if (Start-OcrServer) {
        Start-Process $url
    } else {
        [System.Windows.Forms.MessageBox]::Show("Le serveur OCR n${apostrophe}a pas pu d${eAcute}marrer.", 'OCR Catalogue Monoprix', 'OK', 'Error') | Out-Null
    }
    $openButton.Enabled = $true
    Update-Controls
})

$stopButton.Add_Click({
    $stopButton.Enabled = $false
    if (-not (Stop-OcrServer)) {
        [System.Windows.Forms.MessageBox]::Show("Le serveur ne s${apostrophe}est pas arr${eCircumflex}t${eAcute} correctement.", 'OCR Catalogue Monoprix', 'OK', 'Warning') | Out-Null
    }
    Update-Controls
})

$script:closingApproved = $false
$form.Add_FormClosing({
    param($sender, $eventArgs)
    if (-not $script:closingApproved -and (Test-OcrServer)) {
        $choice = [System.Windows.Forms.MessageBox]::Show(
            "Voulez-vous aussi arr${eCircumflex}ter le serveur OCR ?",
            'Fermer OCR Catalogue Monoprix',
            [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
            [System.Windows.Forms.MessageBoxIcon]::Question
        )
        if ($choice -eq [System.Windows.Forms.DialogResult]::Cancel) {
            $eventArgs.Cancel = $true
            return
        }
        if ($choice -eq [System.Windows.Forms.DialogResult]::Yes) {
            Stop-OcrServer | Out-Null
        }
    }
    $script:closingApproved = $true
})

if (Start-OcrServer) {
    Start-Process $url
}
Update-Controls
[void]$form.ShowDialog()
