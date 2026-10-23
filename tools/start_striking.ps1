param([switch]$Resume, [int]$Ceiling = 32768, [int]$Worlds = 4, [int]$Unroll = 128)
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
$boxingRecord = Join-Path $boxingRuns 'striking_host.json'
if (Test-Path -LiteralPath $boxingRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingRecord -Raw | ConvertFrom-Json
    $boxingProcess = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    if ($boxingProcess -and $boxingProcess.ProcessName -eq 'wsl' -and
        $boxingProcess.StartTime.ToUniversalTime().Ticks -eq ([datetime]$boxingPrevious.started_utc).ToUniversalTime().Ticks) {
        throw "A striking host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_striking.sh', '--ceiling', $Ceiling, '--worlds', $Worlds, '--unroll', $Unroll)
if ($Resume) { $boxingArguments += '--resume' }
$boxingHost = Start-Process wsl.exe -WindowStyle Hidden -WorkingDirectory $boxingProject `
    -ArgumentList $boxingArguments -PassThru `
    -RedirectStandardOutput (Join-Path $boxingRuns 'striking_host.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'striking_host_error.log')
@{ pid=$boxingHost.Id; started_utc=$boxingHost.StartTime.ToUniversalTime().ToString('o');
   ceiling=$Ceiling; mirror='runs/striking-pilot';
   durable_output='$HOME/boxing-runs/striking-pilot-v1' } | ConvertTo-Json | Set-Content -LiteralPath $boxingRecord
Write-Output "Started hidden striking WSL host PID $($boxingHost.Id)."
