param(
    [string]$RemoteHost = "10.21.17.29",
    [string]$RemoteUser = "23039356r",
    [int]$SshPort = 10022,
    [string]$LocalBind = "127.0.0.1",
    [string]$RemoteBind = "127.0.0.1",
    [int]$LocalPort1 = 18001,
    [int]$RemotePort1 = 8001,
    [int]$LocalPort2 = 18002,
    [int]$RemotePort2 = 8002,
    [string]$IdentityFile = "",
    [string]$JumpHost = "",
    [switch]$Single,
    [switch]$Background,
    [switch]$Reconnect
)

$ErrorActionPreference = "Stop"

function Test-CommandExists($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-LocalPortFree($Bind, $Port) {
    $used = Get-NetTCPConnection -LocalAddress $Bind -LocalPort $Port -ErrorAction SilentlyContinue
    if ($used) {
        throw "Local port $Bind`:$Port is already in use. Choose another local port."
    }
}

function Build-SshArgs() {
    $target = if ($RemoteUser) { "$RemoteUser@$RemoteHost" } else { $RemoteHost }
    $forward1 = "$LocalBind`:$LocalPort1`:$RemoteBind`:$RemotePort1"
    $argsList = @(
        "-N",
        "-p", "$SshPort",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=6",
        "-o", "TCPKeepAlive=yes",
        "-L", $forward1
    )
    if (-not $Single) {
        $forward2 = "$LocalBind`:$LocalPort2`:$RemoteBind`:$RemotePort2"
        $argsList += @("-L", $forward2)
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

function Show-Info() {
    $target = if ($RemoteUser) { "$RemoteUser@$RemoteHost" } else { $RemoteHost }
    Write-Host "Starting SSH port forward to $target via ssh port $SshPort" -ForegroundColor Cyan
    Write-Host "Local base URL 1: http://$LocalBind`:$LocalPort1/v1  -> remote $RemoteBind`:$RemotePort1" -ForegroundColor Green
    if (-not $Single) {
        Write-Host "Local base URL 2: http://$LocalBind`:$LocalPort2/v1  -> remote $RemoteBind`:$RemotePort2" -ForegroundColor Green
    }
    Write-Host "Use one of these URLs as cli.py --base-url." -ForegroundColor DarkGray
}

if (-not (Test-CommandExists "ssh")) {
    throw "ssh command was not found. Install OpenSSH Client first."
}

Test-LocalPortFree $LocalBind $LocalPort1
if (-not $Single) {
    Test-LocalPortFree $LocalBind $LocalPort2
}

$argsList = Build-SshArgs
Show-Info

if ($Background) {
    $process = Start-Process -FilePath "ssh" -ArgumentList $argsList -PassThru -WindowStyle Hidden
    Write-Host "SSH tunnel started in background. PID=$($process.Id)" -ForegroundColor Green
    Write-Host "Stop it with: Stop-Process -Id $($process.Id)" -ForegroundColor Yellow
    exit 0
}

Write-Host "Press Ctrl+C to stop the tunnel." -ForegroundColor Yellow
while ($true) {
    & ssh @argsList
    $exitCode = $LASTEXITCODE
    if (-not $Reconnect) {
        exit $exitCode
    }
    Write-Host "SSH tunnel disconnected, exit code: $exitCode" -ForegroundColor Yellow
    Write-Host "Reconnect in 5 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
