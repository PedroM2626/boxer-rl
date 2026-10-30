param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRecord = Join-Path $boxingProject 'runs/gpu_learning_transfer_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'GPU transfer comparison is already running.'
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_gpu_learning_transfer.sh')
if ($Resume) {$boxingArguments += '--resume'}
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList $boxingArguments -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject 'runs/gpu_learning_transfer_host.log') `
    -RedirectStandardError (Join-Path $boxingProject 'runs/gpu_learning_transfer_host_error.log')
@{pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
  diagnostic_output='/home/pedro/boxing-runs/gpu-learning-transfer-v1'; maximum_new_controls=24000} |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden matched GPU transfer host PID $($boxingHost.Id)."
