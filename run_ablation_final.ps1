$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSCommandPath
$pythonExe = $null
$pythonCandidates = @(
    'C:\Users\dell\.pyenv\pyenv-win\versions\3.12.0\python.exe',
    'C:\Users\dell\.pyenv\pyenv-win\shims\python.exe'
)
foreach ($candidate in $pythonCandidates) {
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}
if (-not $pythonExe) {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}
$envFile = Join-Path $projectDir '.env.local'

Write-Host "=== EmoBi 消融实验一键批处理 ===" -ForegroundColor Cyan
Write-Host "项目目录: $projectDir"
Write-Host "Python:    $pythonExe"

$checkPandas = & $pythonExe -c "import pandas" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "所选 Python 缺少 pandas。请先执行: $pythonExe -m pip install pandas"
    exit 1
}
Write-Host "pandas:    OK" -ForegroundColor Green

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\s*$') {
            $key = $Matches[1]; $val = $Matches[2].Trim()
            if (-not (Test-Path "env:$key")) {
                [Environment]::SetEnvironmentVariable($key, $val, 'Process')
                Write-Host "  [env] loaded $key from .env.local" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Warning ".env.local not found, expecting GROQ_API_KEY in environment"
}

if (-not $env:GROQ_API_KEY) {
    Write-Error "GROQ_API_KEY is not set. Please create .env.local with: GROQ_API_KEY=your_key"
    exit 1
}

$experiments = @(
    @{ Profile='wo_feature'; Desc='关闭情感词典与概念特征' },
    @{ Profile='plus_e';     Desc='仅开启情感词典注入' },
    @{ Profile='plus_c';     Desc='仅开启概念特征注入' },
    @{ Profile='full';       Desc='完整方案 --- 最终对照' }
)

$logDir = Join-Path $projectDir 'results\tables'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ('ablation_live_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $logPath -Append

Write-Host "共 $($experiments.Count) 组消融实验" -ForegroundColor Yellow

$failed = @()
$completed = @()
$startTime = Get-Date

for ($idx = 0; $idx -lt $experiments.Count; $idx++) {
    $exp = $experiments[$idx]
    $num = $idx + 1
    $runName = "ablation_$($exp.Profile)"
    $resultsDir = Join-Path $projectDir 'results\runs' $runName
    $stageFinal = Join-Path $resultsDir 'stage_final.csv'

    Write-Host "`n[$num/$($experiments.Count)] profile=$($exp.Profile)  $($exp.Desc)" -ForegroundColor Green

    if (Test-Path $stageFinal) {
        Write-Host "  [skip] $stageFinal already exists" -ForegroundColor DarkYellow
        $completed += $exp.Profile
        continue
    }

    $elapsed = ((Get-Date) - $startTime).ToString('hh\:mm\:ss')
    Write-Host "  [run]  已耗时 $elapsed" -ForegroundColor Gray

    $cmdArgs = @(
        $projectDir + '\main.py',
        '--input-name', 'hypo-l.csv',
        '--ablation-profile', $exp.Profile,
        '--run-name', $runName,
        '--resume-enable', '1',
        '--checkpoint-interval', '5'
    )

    $proc = Start-Process -FilePath $pythonExe -ArgumentList $cmdArgs -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -eq 0) {
        if (Test-Path $stageFinal) {
            Write-Host "  [ok]   $($exp.Profile) completed" -ForegroundColor Green
            $completed += $exp.Profile
        } else {
            Write-Host "  [warn] exit OK but $stageFinal missing" -ForegroundColor Yellow
            $failed += $exp.Profile
        }
    } else {
        Write-Host "  [fail] exit code $($proc.ExitCode)" -ForegroundColor Red
        $failed += $exp.Profile
    }
}

Write-Host "`n=== 汇总 ===" -ForegroundColor Cyan
Write-Host "已完成: $($completed -join ', ')"
if ($failed.Count -gt 0) {
    Write-Host "失败:   $($failed -join ', ')" -ForegroundColor Red
}

$totalElapsed = ((Get-Date) - $startTime).ToString('hh\:mm\:ss')
Write-Host "总耗时: $totalElapsed" -ForegroundColor Cyan

Write-Host "`n=== 自动汇总指标 ===" -ForegroundColor Cyan
$collectArgs = @(
    $projectDir + '\collect_final_metrics.py',
    '--root-dir', (Join-Path $projectDir 'results\runs'),
    '--out-dir', $logDir
)
Start-Process -FilePath $pythonExe -ArgumentList $collectArgs -NoNewWindow -Wait

Stop-Transcript

$metricsPath = Join-Path $logDir 'main_metrics.csv'
if (Test-Path $metricsPath) {
    Write-Host "`n=== 结果总表 ===" -ForegroundColor Cyan
    Get-Content $metricsPath -Encoding UTF8
    Write-Host "`n完整日志: $logPath" -ForegroundColor Gray
}
