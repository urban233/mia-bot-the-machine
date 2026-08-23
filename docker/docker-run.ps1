param (
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassthroughArgs
)

# Resolve repository root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Get-Item "$ScriptDir\..").FullName
$ImageName = "mia-bot-train:latest"

# Check if image exists, build if not
$imageExists = docker images -q $ImageName
if (-not $imageExists) {
    Write-Host "[+] Building Docker image $ImageName..." -ForegroundColor Cyan
    docker build -t $ImageName -f "$ScriptDir\Dockerfile" "$RepoRoot"
}

# Ensure host data directory exists
$DataPath = Join-Path $RepoRoot "data"
if (-not (Test-Path $DataPath)) {
    New-Item -ItemType Directory -Path $DataPath | Out-Null
}

Write-Host "[+] Launching MIA-BOT training container with GPU acceleration..." -ForegroundColor Green
$cmdArgs = @(
    "run", "--rm", "-it",
    "--gpus", "all",
    "--ipc", "host",
    "--shm-size", "2gb",
    "-v", "${DataPath}:/app/data",
    $ImageName
)
if ($PassthroughArgs) {
    $cmdArgs += $PassthroughArgs
}

& docker @cmdArgs

