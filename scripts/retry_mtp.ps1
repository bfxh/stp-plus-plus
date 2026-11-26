# Retry loop for the missing MTP / gap-test arms.
# A run that fails (e.g. OOM) leaves no run.json, so the next pass picks it up again.
$root = $PSScriptRoot | Split-Path -Parent
$py   = if ($env:STPP_PYTHON) { $env:STPP_PYTHON } else { "python" }
$arms = @("ce33", "ce33_mtp2", "mtp2", "mtp3", "ce25_mtp3", "mtp3_l", "ce25_mtp3_l")

for ($pass = 1; $pass -le 5; $pass++) {
    "=== pass $pass  $(Get-Date -Format 'HH:mm') ==="
    $a = @((Join-Path $root "scripts\run_pretrain.py"), "--arms") + $arms +
         @("--seeds", "82", "23", "37", "--steps", "120", "--bs", "4",
           "--out-root", "runs\pretrain")
    & $py @a 2>&1 | Select-String "ppl=|fail|pass"
    Start-Sleep -Seconds 45
}
"done $(Get-Date -Format 'HH:mm')"
