$ErrorActionPreference = 'Stop'

# Native Windows counterpart of scripts/release-preflight.sh. It mirrors what
# .github/workflows/release.yml does for x86_64-pc-windows-gnu: the exe lives in
# target\x86_64-pc-windows-gnu\release, WebView2Loader.dll is found anywhere under
# target\ (webview2-com-sys\out\x64), and sensor_bridge is published with the
# .NET 10 SDK (win-x64).

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

$repo = Join-Path $PSScriptRoot '..'
$targetRoot = Join-Path $repo 'target'
$release = Join-Path $targetRoot 'x86_64-pc-windows-gnu\release'
foreach ($binary in @('local-llm-foundry.exe')) {
    $path = Join-Path $release $binary
    if (-not (Test-Path -LiteralPath $path)) {
        throw "FAIL: missing release binary $path (build with: cargo build --release --target x86_64-pc-windows-gnu)"
    }
}

$webview = Get-ChildItem $targetRoot -Filter WebView2Loader.dll -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'webview2-com-sys' -and $_.FullName -match '\\x64\\' } | Select-Object -First 1
if (-not $webview) {
    throw 'FAIL: x64 WebView2Loader.dll not found under target\ (webview2-com-sys build output)'
}

dotnet restore (Join-Path $repo 'sensor_bridge\sensor_bridge.csproj')
if ($LASTEXITCODE -ne 0) { throw 'FAIL: sensor bridge restore failed' }
dotnet build (Join-Path $repo 'sensor_bridge\sensor_bridge.csproj') -c Release -r win-x64 --no-restore
if ($LASTEXITCODE -ne 0) { throw 'FAIL: sensor bridge build failed' }
cargo metadata --no-deps --format-version 1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'FAIL: cargo metadata failed' }

Write-Host 'PASS: native Windows release preflight'
Write-Host "PASS: .NET SDK $((($sdk10 | Select-Object -First 1).ToString()).Trim())"
Write-Host "PASS: WebView2 loader $($webview.FullName)"
