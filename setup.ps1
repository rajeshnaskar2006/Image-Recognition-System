$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

function Test-CompatiblePython([string] $Executable, [string[]] $Arguments = @()) {
    try {
        & $Executable @Arguments -c "import sys, struct; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8 else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        $candidate = $null
        $candidateArgs = @()
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            foreach ($version in @('-3.12', '-3.11', '-3.10')) {
                if (Test-CompatiblePython $launcher.Source @($version)) {
                    $candidate = $launcher.Source
                    $candidateArgs = @($version)
                    break
                }
            }
        }
        if (-not $candidate) {
            $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($onPath -and (Test-CompatiblePython $onPath.Source)) { $candidate = $onPath.Source }
        }
        if (-not $candidate) {
            # Codex includes a compatible runtime on this computer; use it if present.
            $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
            if ((Test-Path -LiteralPath $bundledPython) -and (Test-CompatiblePython $bundledPython)) {
                $candidate = $bundledPython
            }
        }
        if (-not $candidate) {
            throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/ and run setup.bat again.'
        }
        Write-Host 'Creating a private Python environment for this project...'
        & $candidate @candidateArgs -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    if (-not (Test-CompatiblePython $projectPython)) {
        throw 'The existing .venv does not have a compatible Python. Rename .venv and run setup.bat again with Python 3.12 installed.'
    }
    & $projectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip. Check your internet connection.' }
    & $projectPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the error above and your internet connection.' }
    & $projectPython (Join-Path $PSScriptRoot 'prepare_model.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Check your internet connection and the error above.' }
    Write-Host ''
    Write-Host 'Setup complete! Double-click start.bat and open http://127.0.0.1:5000' -ForegroundColor Green
    exit 0
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
