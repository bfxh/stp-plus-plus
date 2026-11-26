# 跑「缺口测试」的缺失臂；失败的 run 不会写 run.json，下一 pass 自动补。
#   .\scripts\retry_mtp.ps1                       # 全部 7 臂（约 18 个 run）
#   $env:STPP_CORE_ONLY="1"; .\scripts\retry_mtp.ps1   # 只跑决定性三臂（9 个 run）
$root = $PSScriptRoot | Split-Path -Parent
$py   = if ($env:STPP_PYTHON) { $env:STPP_PYTHON } else { "python" }

$core  = @("ce33", "ce33_mtp2", "mtp2")
$extra = @("mtp3", "ce25_mtp3", "mtp3_l", "ce25_mtp3_l")
$arms  = if ($env:STPP_CORE_ONLY -eq "1") { $core } else { $core + $extra }

$extraArgs = @()
if ($env:STPP_LOAD_8BIT -eq "1") { $extraArgs += "--load_8bit" }

for ($pass = 1; $pass -le 3; $pass++) {
    "=== pass $pass  $(Get-Date -Format 'HH:mm') ==="
    $a = @((Join-Path $root "scripts\run_pretrain.py"), "--arms") + $arms +
         @("--seeds", "82", "23", "37", "--steps", "120", "--bs", "4",
           "--out-root", "runs\pretrain") + $extraArgs
    & $py @a 2>&1 | Select-String "ppl=|fail"
    Start-Sleep -Seconds 20
}
"done $(Get-Date -Format 'HH:mm')"
& $py (Join-Path $root "scripts\agg_pretrain.py")
