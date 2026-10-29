$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRecord = Join-Path $boxingProject 'runs/gpu_batched_integration_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'GPU integration host is already running.'
    }
}
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_gpu_batched_integration.sh') -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject 'runs/gpu_batched_integration_host.log') `
    -RedirectStandardError (Join-Path $boxingProject 'runs/gpu_batched_integration_host_error.log')
@{pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
  diagnostic_output='/home/pedro/boxing-runs/gpu-batched-integration-v1'; maximum_new_ppo_steps=1024} |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden GPU integration host PID $($boxingHost.Id)."
