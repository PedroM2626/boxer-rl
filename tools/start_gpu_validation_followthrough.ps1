param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRecord = Join-Path $boxingProject 'runs/gpu_validation_followthrough_host.json'
$boxingComparison = (& wsl.exe -e bash -lc 'cat /home/pedro/boxing-runs/gpu-closed-loop-v1/state.json') | ConvertFrom-Json
if ($boxingComparison.status -notin @('running', 'completed')) {
    throw 'Wait for the reviewed comparison resume to enter running state before starting the coordinator.'
}
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'GPU validation coordinator is already running.'
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_gpu_validation_followthrough.sh')
if ($Resume) { $boxingArguments += '--resume' }
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList $boxingArguments -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject 'runs/gpu_validation_followthrough_host.log') `
    -RedirectStandardError (Join-Path $boxingProject 'runs/gpu_validation_followthrough_host_error.log')
@{pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
  diagnostic_output='/home/pedro/boxing-runs/gpu-validation-followthrough'} |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden validation coordinator PID $($boxingHost.Id)."
