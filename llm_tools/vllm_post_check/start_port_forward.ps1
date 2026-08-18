param(
    [string]$RemoteHost = "10.21.17.29",
    [string]$RemoteUser = "23039356r",
    [int]$SshPort = 10022,

    [string]$LocalBind = "127.0.0.1",
    [string]$RemoteBind = "127.0.0.1",

    [string[]]$PortMappings = @(
        "18001:8001",
        "18002:8002",
        "18010:8010"
    ),

    [string]$IdentityFile = "",
    [string]$JumpHost = "",

    [switch]$Background,
    [switch]$Reconnect
)

$ErrorActionPreference = "Stop"


function Test-CommandExists {
    param([string]$Name)

    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}


function Parse-PortMapping {
    param([string]$Mapping)

    $parts = $Mapping.Split(":")

    if ($parts.Count -ne 2) {
        throw "Invalid mapping '$Mapping'. Expected format: LocalPort:RemotePort"
    }

    $localPort = 0
    $remotePort = 0

    if (
        -not [int]::TryParse($parts[0], [ref]$localPort) -or
        -not [int]::TryParse($parts[1], [ref]$remotePort)
    ) {
        throw "Invalid mapping '$Mapping'. Ports must be integers."
    }

    if ($localPort -lt 1 -or $localPort -gt 65535) {
        throw "Invalid local port: $localPort"
    }

    if ($remotePort -lt 1 -or $remotePort -gt 65535) {
        throw "Invalid remote port: $remotePort"
    }

    return @{
        LocalPort  = $localPort
        RemotePort = $remotePort
    }
}


function Test-LocalPortFree {
    param(
        [string]$Bind,
        [int]$Port
    )

    $used = Get-NetTCPConnection `
        -LocalAddress $Bind `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue

    if ($used) {
        throw "Local port $Bind`:$Port is already in use."
    }
}


function Build-SshArgs {
    $target = if ($RemoteUser) {
        "$RemoteUser@$RemoteHost"
    }
    else {
        $RemoteHost
    }

    $argsList = @(
        "-N",
        "-p", "$SshPort",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=6",
        "-o", "TCPKeepAlive=yes"
    )

    foreach ($mappingText in $PortMappings) {
        $mapping = Parse-PortMapping $mappingText

        $forward = (
            "$LocalBind`:$($mapping.LocalPort)" +
            "`:$RemoteBind`:$($mapping.RemotePort)"
        )

        $argsList += @("-L", $forward)
    }

    if ($IdentityFile) {
        $argsList += @("-i", $IdentityFile)
    }

    if ($JumpHost) {
        $argsList += @("-J", $JumpHost)
    }

    $argsList += $target

    return $argsList
}


function Show-Info {
    $target = if ($RemoteUser) {
        "$RemoteUser@$RemoteHost"
    }
    else {
        $RemoteHost
    }

    Write-Host `
        "Starting SSH port forwarding to $target via SSH port $SshPort" `
        -ForegroundColor Cyan

    foreach ($mappingText in $PortMappings) {
        $mapping = Parse-PortMapping $mappingText

        Write-Host (
            "http://$LocalBind`:$($mapping.LocalPort)/v1" +
            " -> $RemoteBind`:$($mapping.RemotePort)"
        ) -ForegroundColor Green
    }
}


if (-not (Test-CommandExists "ssh")) {
    throw "ssh command was not found. Install OpenSSH Client first."
}


# 检查重复和占用
$seenLocalPorts = @{}

foreach ($mappingText in $PortMappings) {
    $mapping = Parse-PortMapping $mappingText
    $localPort = $mapping.LocalPort

    if ($seenLocalPorts.ContainsKey($localPort)) {
        throw "Duplicate local port: $localPort"
    }

    $seenLocalPorts[$localPort] = $true
    Test-LocalPortFree -Bind $LocalBind -Port $localPort
}


$argsList = Build-SshArgs
Show-Info


if ($Background) {
    $process = Start-Process `
        -FilePath "ssh" `
        -ArgumentList $argsList `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "SSH tunnel started. PID=$($process.Id)" `
        -ForegroundColor Green

    Write-Host "Stop it with: Stop-Process -Id $($process.Id)" `
        -ForegroundColor Yellow

    exit 0
}


Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor Yellow

while ($true) {
    & ssh @argsList
    $exitCode = $LASTEXITCODE

    if (-not $Reconnect) {
        exit $exitCode
    }

    Write-Host `
        "SSH tunnel disconnected, exit code: $exitCode" `
        -ForegroundColor Yellow

    Start-Sleep -Seconds 5
}