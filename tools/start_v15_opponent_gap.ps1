$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRecord = Join-Path $boxingProject 'runs/v15_opponent_gap_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'The opponent-gap diagnostic host is already running.'
    }
}
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_v15_opponent_gap.sh') -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject 'runs/v15_opponent_gap_host.log') `
    -RedirectStandardError (Join-Path $boxingProject 'runs/v15_opponent_gap_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output='/home/pedro/boxing-runs/striking-v15-opponent-gap'; maximum_new_controls=8000 } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden opponent-gap diagnostic host PID $($boxingHost.Id)."
