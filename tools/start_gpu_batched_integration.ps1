param([switch]$Scaled)
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingName = if ($Scaled) {'gpu_batched_scaling'} else {'gpu_batched_integration'}
$boxingRecord = Join-Path $boxingProject "runs/${boxingName}_host.json"
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'GPU integration host is already running.'
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_gpu_batched_integration.sh')
if ($Scaled) {$boxingArguments += '--scaled'}
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList $boxingArguments -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject "runs/${boxingName}_host.log") `
    -RedirectStandardError (Join-Path $boxingProject "runs/${boxingName}_host_error.log")
@{pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
  diagnostic_output=$(if ($Scaled) {'/home/pedro/boxing-runs/gpu-batched-scaling-v1'} else {'/home/pedro/boxing-runs/gpu-batched-integration-v1'});
  maximum_new_ppo_steps=$(if ($Scaled) {16384} else {1024})} |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden GPU integration host PID $($boxingHost.Id)."
