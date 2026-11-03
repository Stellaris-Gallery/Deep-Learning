<#
.SYNOPSIS
Build the isolated COMP395 Windows environment from the checked-in lockfile.
.EXAMPLE
.\Tools\setup_environment.ps1 -ExpectedGpuCount 2
.EXAMPLE
.\Tools\setup_environment.ps1 -PythonExe C:\Python314\python.exe -ExpectedGpuCount 1
#>
[CmdletBinding()]
param(
    [string]$PythonExe,
    [ValidateRange(1, 16)][int]$ExpectedGpuCount = 1,
    [switch]$SkipKernelRegistration
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$envDirectory = Join-Path $repoRoot '.venv'
$envPython = Join-Path $envDirectory 'Scripts\python.exe'
$lockFile = Join-Path $repoRoot 'environment\requirements.lock.txt'
$runtime = Join-Path $repoRoot '.runtime'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')" }
}

if ($env:OS -ne 'Windows_NT') { throw 'This lock targets Windows x86-64. See environment/README.md.' }
if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) { throw "Missing lockfile: $lockFile" }
if (-not $PythonExe) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install 64-bit Python 3.14 or pass -PythonExe with its full path.' }
    $PythonExe = & py -3.14 -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.14 was not found. Install it or pass -PythonExe.' }
}
$PythonExe = (Get-Command $PythonExe -ErrorAction Stop).Source
$versionCheck = "import sys; assert sys.version_info[:2] == (3,14) and sys.maxsize > 2**32, '64-bit Python 3.14 required'; print(sys.version)"
Invoke-Checked $PythonExe @('-c', $versionCheck)

if (Test-Path -LiteralPath $envDirectory) {
    if (-not (Test-Path -LiteralPath $envPython -PathType Leaf)) { throw '.venv exists but is not a Windows Python environment. Rename it before retrying.' }
    Invoke-Checked $envPython @('-c', $versionCheck)
    $config = Get-Content -Raw -LiteralPath (Join-Path $envDirectory 'pyvenv.cfg')
    if ($config -notmatch '(?im)^include-system-site-packages\s*=\s*false\s*$') { throw 'Refusing to reuse an environment that inherits system packages. Rename .venv and retry.' }
} else {
    Invoke-Checked $PythonExe @('-m', 'venv', $envDirectory)
}

New-Item -ItemType Directory -Force -Path $runtime | Out-Null
# Ignore inherited Python and pip settings for this process only.
$savedEnvironment = @{}
$environmentNames = @('PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PIP_CONFIG_FILE', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_TARGET', 'PIP_USER', 'PIP_REQUIRE_VIRTUALENV')
foreach ($name in $environmentNames) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    foreach ($name in $environmentNames) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    $env:PYTHONNOUSERSITE = '1'
    $env:PIP_CONFIG_FILE = 'NUL'
    Write-Host 'Installing locked dependencies. The CUDA wheel is a large download.'
    Invoke-Checked $envPython @('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', '--find-links', (Join-Path $repoRoot 'environment\wheels'), '--only-binary', 'gensim', '--require-hashes', '--no-cache-dir', '--disable-pip-version-check', '-r', $lockFile)
    Invoke-Checked $envPython @('-m', 'pip', 'check')
    Invoke-Checked $envPython @((Join-Path $PSScriptRoot 'check_environment.py'), '--expected-gpus', "$ExpectedGpuCount", '--report', (Join-Path $runtime 'environment-report.json'))
    if (-not $SkipKernelRegistration) {
        Invoke-Checked $envPython @('-m', 'ipykernel', 'install', '--prefix', $envDirectory, '--name', 'comp395', '--display-name', 'COMP395 (Python 3.14, CUDA 13.0)')
    }
} finally {
    foreach ($name in $environmentNames) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
}
Write-Host "Environment ready: $envPython"
Write-Host 'Select this interpreter/kernel in VS Code. No notebooks have been executed.'
