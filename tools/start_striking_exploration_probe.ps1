$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
$boxingRecord = Join-Path $boxingRuns 'striking_exploration_probe_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw "A exploration-probe host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingOutput = '/home/pedro/boxing-runs/striking-exploration-probe-1441792'
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_striking_exploration_probe.sh', '--output', $boxingOutput) `
    -PassThru -RedirectStandardOutput (Join-Path $boxingRuns 'striking_exploration_probe_host.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'striking_exploration_probe_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output=$boxingOutput; maximum_world_controls=32000; optimization_transitions=0; main_consumed_steps=1048576 } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden bounded exploration-probe WSL host PID $($boxingHost.Id)."
