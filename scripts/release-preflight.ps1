$ErrorActionPreference = 'Stop'

Write-Host 'Running native Windows release preflight checks...'

foreach ($tool in @('cargo', 'rustup', 'dotnet')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "FAIL: $tool not found"
    }
}

$sdk10 = dotnet --list-sdks | Where-Object { $_ -match '^10\.' }
if (-not $sdk10) {
    throw 'FAIL: .NET 10 SDK is required for sensor_bridge'
}

$targets = rustup target list --installed
if (-not ($targets -contains 'x86_64-pc-windows-gnu')) {
    throw 'FAIL: rustup target x86_64-pc-windows-gnu is not installed'
}

$release = Join-Path $PSScriptRoot '..\target\release'
foreach ($binary in @('local-llm-foundry.exe')) {
    $path = Join-Path $release $binary
    if (-not (Test-Path -LiteralPath $path)) {
        throw "FAIL: missing release binary $path"
    }
}

$webview = Get-ChildItem (Join-Path $release 'build') -Filter WebView2Loader.dll -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\x64\\' } | Select-Object -First 1
if (-not $webview) {
    throw 'FAIL: x64 WebView2Loader.dll not found in release build output'
}

dotnet restore (Join-Path $PSScriptRoot '..\sensor_bridge\sensor_bridge.csproj')
if ($LASTEXITCODE -ne 0) { throw 'FAIL: sensor bridge restore failed' }
dotnet build (Join-Path $PSScriptRoot '..\sensor_bridge\sensor_bridge.csproj') -c Release -r win-x64 --no-restore
if ($LASTEXITCODE -ne 0) { throw 'FAIL: sensor bridge build failed' }
cargo metadata --no-deps --format-version 1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'FAIL: cargo metadata failed' }

Write-Host 'PASS: native Windows release preflight'
Write-Host "PASS: .NET SDK $((($sdk10 | Select-Object -First 1).ToString()).Trim())"
Write-Host "PASS: WebView2 loader $($webview.FullName)"
