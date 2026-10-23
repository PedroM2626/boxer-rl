$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
$boxingRecord = Join-Path $boxingRuns 'striking_mixed_confirmation_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw "A mixed-opponent continuation host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingOutput = '/home/pedro/boxing-runs/striking-mixed-confirm-1851392'
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_striking_mixed_confirm_continue.sh', '--output', $boxingOutput) `
    -PassThru -RedirectStandardOutput (Join-Path $boxingRuns 'striking_mixed_confirmation_host.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'striking_mixed_confirmation_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output=$boxingOutput; maximum_evaluation_world_controls=48000;
   maximum_new_ppo_steps=16384; main_consumed_steps=1048576 } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden bounded mixed-opponent continuation WSL host PID $($boxingHost.Id)."
