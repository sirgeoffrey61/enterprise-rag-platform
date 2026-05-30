# Start Redis (Docker preferred, or local infra/redis binary)
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    $candidates = @(
        "${env:ProgramFiles}\Docker\Docker\resources\bin\docker.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $docker = $candidate; break }
    }
}

if ($docker) {
    $dockerExe = if ($docker -is [string]) { $docker } else { $docker.Source }
    $existing = & $dockerExe ps -a --filter "name=redis" --format "{{.Names}}" 2>$null
    if ($existing -eq "redis") {
        & $dockerExe start redis
    } else {
        & $dockerExe run -d --name redis -p 6379:6379 redis:alpine
    }
    Write-Host "Redis (Docker) listening on localhost:6379"
    exit 0
}

$redisServer = Join-Path $PSScriptRoot "redis\redis-server.exe"
if (Test-Path $redisServer) {
    Start-Process -FilePath $redisServer -ArgumentList "--port", "6379" -WindowStyle Hidden
    Write-Host "Redis (local binary) listening on localhost:6379"
    exit 0
}

Write-Error "Neither Docker nor infra/redis/redis-server.exe found."
