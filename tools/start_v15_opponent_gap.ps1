param([switch]$NavigationAblation)
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingStem = if ($NavigationAblation) { 'v15_navigation_ablation' } else { 'v15_opponent_gap' }
$boxingRecord = Join-Path $boxingProject "runs/$($boxingStem)_host.json"
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw 'The opponent-gap diagnostic host is already running.'
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_v15_opponent_gap.sh')
if ($NavigationAblation) { $boxingArguments += '--navigation-ablation' }
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList $boxingArguments -PassThru `
    -RedirectStandardOutput (Join-Path $boxingProject "runs/$($boxingStem)_host.log") `
    -RedirectStandardError (Join-Path $boxingProject "runs/$($boxingStem)_host_error.log")
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   diagnostic_output=$(if ($NavigationAblation) { '/home/pedro/boxing-runs/striking-v15-navigation-ablation' } else { '/home/pedro/boxing-runs/striking-v15-opponent-gap' });
   maximum_new_controls=$(if ($NavigationAblation) { 4000 } else { 8000 }) } |
    ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden opponent-gap diagnostic host PID $($boxingHost.Id)."
