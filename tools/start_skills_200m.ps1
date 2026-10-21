param([switch]$Publish)
# A persistent hidden Windows process keeps the foreground WSL supervisor alive.
$ErrorActionPreference = 'Stop'
$boxingProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$boxingRuns = Join-Path $boxingProject 'runs'
New-Item -ItemType Directory -Path $boxingRuns -Force | Out-Null
$boxingHostRecord = Join-Path $boxingRuns 'skills_200m_host.json'
if (Test-Path -LiteralPath $boxingHostRecord) {
    $boxingPrevious = Get-Content -LiteralPath $boxingHostRecord -Raw | ConvertFrom-Json
    $boxingExisting = Get-Process -Id $boxingPrevious.pid -ErrorAction SilentlyContinue
    # PowerShell may deserialize ISO JSON strings as DateTime. Compare UTC
    # ticks rather than a formatted string against a deserialized object.
    $boxingRecordedStart = [datetime]$boxingPrevious.started_utc
    if ($boxingExisting -and $boxingExisting.ProcessName -eq 'wsl' -and
        $boxingExisting.StartTime.ToUniversalTime().Ticks -eq $boxingRecordedStart.ToUniversalTime().Ticks) {
        throw "A training WSL host is already running: PID $($boxingPrevious.pid)."
    }
}
$boxingArguments = @('-e', 'bash', 'tools/run_skills_200m.sh')
if ($Publish) { $boxingArguments += '--publish' }
$boxingHost = Start-Process -FilePath 'wsl.exe' -ArgumentList $boxingArguments `
    -WorkingDirectory $boxingProject -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $boxingRuns 'skills_200m_supervisor.log') `
    -RedirectStandardError (Join-Path $boxingRuns 'skills_200m_host_error.log')
@{
    pid = $boxingHost.Id
    started_utc = $boxingHost.StartTime.ToUniversalTime().ToString('o')
    publishing = [bool]$Publish
} | ConvertTo-Json | Set-Content -LiteralPath $boxingHostRecord
Write-Output "Started hidden WSL host PID $($boxingHost.Id). Verify runs/skills_200m/job_state.json and progress.jsonl."
