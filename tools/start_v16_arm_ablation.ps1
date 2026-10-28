$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRecord = Join-Path $boxingProject 'runs/v16_arm_ablation_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'The arm-ablation host is already running.'
    }
}
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList @('-e', 'bash', 'tools/run_v16_arm_ablation.sh') -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject 'runs/v16_arm_ablation_host.log') `
    -RedirectStandardError (Join-Path $boxingProject 'runs/v16_arm_ablation_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output='/home/pedro/boxing-runs/striking-v16-arm-ablation'; maximum_new_controls=4000 } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden arm-ablation diagnostic host PID $($boxingHost.Id)."
