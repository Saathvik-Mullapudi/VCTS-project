param (
    [string]$Video = "data/videos/vid.mp4",
    [int]$TimeoutSecs = 15
)

Write-Host "Running smoke test on $Video for $TimeoutSecs seconds..." -ForegroundColor Cyan

# We run main.py in the background, collect output, and force-kill it after timeout
$process = Start-Process python -ArgumentList "main.py --video $Video" -PassThru -RedirectStandardOutput "smoke_output.log" -RedirectStandardError "smoke_error.log" -WindowStyle Hidden

Start-Sleep -Seconds $TimeoutSecs

if (!$process.HasExited) {
    Write-Host "Timeout reached, stopping process safely..." -ForegroundColor Yellow
    Stop-Process -Id $process.Id -Force
}

$output = Get-Content "smoke_output.log" -ErrorAction SilentlyContinue
$errors = Get-Content "smoke_error.log" -ErrorAction SilentlyContinue

Write-Host "`n=== SMOKE TEST SUMMARY ===" -ForegroundColor Cyan
if ($errors -match "(?i)Traceback|Exception|RuntimeError|TypeError|ValueError|ImportError") {
    Write-Host "FAILED: Errors detected in the logs!" -ForegroundColor Red
    $errors | Select-String -Pattern "(?i)Traceback|Exception|RuntimeError|TypeError|ValueError|ImportError" -Context 2,2 | Out-Host
} else {
    Write-Host "PASS: No Python exceptions detected during runtime." -ForegroundColor Green
}

$crossings = $output | Select-String -Pattern "Crossings:" | Select-Object -Last 1
if ($crossings) {
    Write-Host "System achieved crossings: $($crossings.Line.Trim())" -ForegroundColor Green
}

Write-Host "Outputs saved to smoke_output.log and smoke_error.log" -ForegroundColor DarkGray
