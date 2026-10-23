param([int]$Steps = 131072, [ValidateSet('reach', 'imitation')][string]$Candidate = 'reach')
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
$boxingRecord = Join-Path $boxingRuns 'striking_comparison_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw "A comparison host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingOutput = if ($Candidate -eq 'imitation') { '/home/pedro/boxing-runs/striking-imitation-compare-1048576' } else { '/home/pedro/boxing-runs/striking-reward-compare-1048576' }
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_striking_reward_compare.sh', '--output', $boxingOutput, '--steps', $Steps, '--candidate', $Candidate) `
    -PassThru -RedirectStandardOutput (Join-Path $boxingRuns 'striking_comparison_host.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'striking_comparison_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output=$boxingOutput; steps_per_arm=$Steps; candidate=$Candidate; main_consumed_steps=1048576 } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden diagnostic comparison WSL host PID $($boxingHost.Id)."
