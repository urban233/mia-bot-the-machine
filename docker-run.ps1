param (
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassthroughArgs
)

$ImageName = "mia-bot-train:latest"

# Check if image exists, build if not
$imageExists = docker images -q $ImageName
if (-not $imageExists) {
    Write-Host "[+] Building Docker image $ImageName..." -ForegroundColor Cyan
    docker build -t $ImageName .
}

# Ensure data directory exists on host
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

Write-Host "[+] Launching mia-bot training container..." -ForegroundColor Green
$cmdArgs = @("run", "--rm", "-it", "--gpus", "all", "--shm-size", "2gb", "-v", "$((Get-Location).Path)/data:/app/data", $ImageName)
if ($PassthroughArgs) {
    $cmdArgs += $PassthroughArgs
}

& docker @cmdArgs

