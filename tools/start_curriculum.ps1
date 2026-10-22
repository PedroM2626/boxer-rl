param([switch]$Publish, [switch]$WarmStart, [int]$MaxTotalSteps = 200000000, [int]$GateEvery = 5000000, [int]$Worlds = 1024, [string]$RestartFrom)
# A hidden persistent Windows host keeps foreground WSL training alive.
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
New-Item -ItemType Directory -Path $boxingRuns -Force | Out-Null
$boxingHostRecord = Join-Path $boxingRuns 'curriculum_host.json'
if (Test-Path -LiteralPath $boxingHostRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingHostRecord -Raw | ConvertFrom-Json
    $boxingExisting = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    $boxingRecordedStart = [datetime]$boxingPrevious.started_utc
    if ($boxingExisting -and $boxingExisting.ProcessName -eq 'wsl' -and
        $boxingExisting.StartTime.ToUniversalTime().Ticks -eq $boxingRecordedStart.ToUniversalTime().Ticks) {
        throw "A curriculum WSL host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_curriculum.sh', '--max-total-steps', $MaxTotalSteps, '--gate-every', $GateEvery, '--worlds', $Worlds)
if ($Publish) { $boxingArguments += '--publish' }
if ($WarmStart) { $boxingArguments += '--warm-start' }
if ($RestartFrom) { $boxingArguments += @('--restart-from', $RestartFrom) }
$boxingHost = Start-Process -FilePath 'wsl.exe' -ArgumentList $boxingArguments `
    -WorkingDirectory $boxingProject -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $boxingRuns 'curriculum_supervisor.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'curriculum_host_error.log')
@{
    pid = $boxingHost.Id
    started_utc = $boxingHost.StartTime.ToUniversalTime().ToString('o')
    publishing = [bool]$Publish
} | ConvertTo-Json | Set-Content -LiteralPath $boxingHostRecord
Write-Output "Started hidden WSL host PID $($boxingHost.Id). Inspect runs/curriculum/state.json and the active level's progress.json."
