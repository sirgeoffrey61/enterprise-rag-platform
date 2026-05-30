# Start Qdrant via Docker (Task 4)
$storagePath = "E:\Enterpirse Rag folder\enterprise-rag-platform\data\qdrant_storage"
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    $candidates = @(
        "${env:ProgramFiles}\Docker\Docker\resources\bin\docker.exe",
        "$env:LOCALAPPDATA\Programs\Docker\Docker\resources\bin\docker.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $docker = $candidate; break }
    }
}
if (-not $docker) {
    Write-Error "Docker not found. Install Docker Desktop and ensure 'docker' is on PATH."
    exit 1
}

$dockerExe = if ($docker -is [string]) { $docker } else { $docker.Source }
$existing = & $dockerExe ps -a --filter "name=qdrant" --format "{{.Names}}" 2>$null
if ($existing -eq "qdrant") {
    & $dockerExe start qdrant
} else {
    & $dockerExe run -d --name qdrant -p 6333:6333 -p 6334:6334 `
        -v "${storagePath}:/qdrant/storage" `
        qdrant/qdrant
}
Write-Host "Qdrant should be available at http://localhost:6333"
