param(
    [int]$IntervalSeconds = 300,
    [int]$MaxMinutes = 30,
    [string]$Account = 'code12ab@gmail.com',
    [string]$RccPath = 'D:\Programs\miniconda3\envs\stock\Scripts\rcc.exe'
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 10 -or $MaxMinutes -lt 1) { throw 'Invalid polling interval or time limit.' }
if (-not (Test-Path -LiteralPath $RccPath)) { throw "RCC not found: $RccPath" }
$env:PYTHONIOENCODING = 'utf-8'
$deadline = (Get-Date).AddMinutes($MaxMinutes)
do {
    $raw = & $RccPath -A $Account instance status --json
    if ($LASTEXITCODE -ne 0) { throw 'RCC status failed; no create or delete action was taken.' }
    $state = ($raw -join "`n") | ConvertFrom-Json
    Write-Host ('{0:yyyy-MM-dd HH:mm:ss} status={1} reason={2}' -f (Get-Date), $state.status, $state.reason)
    if ($state.ready -or $state.status -eq 'ready') {
        Write-Host 'Instance ready. It may now consume GPU credits.'
        & $RccPath -A $Account instance ssh --print
        return
    }
    if ($state.status -notin @('pending', 'allocating', 'creating', 'starting', 'provisioning')) {
        Write-Host $state.message
        return
    }
    $remaining = [int][Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds)
    if ($remaining -le 0) { break }
    Start-Sleep -Seconds ([Math]::Min($IntervalSeconds, $remaining))
} while ((Get-Date) -lt $deadline)
Write-Host 'Monitoring time limit reached. The queued instance was NOT canceled; use rcc instance status to check it.'
